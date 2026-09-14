from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from scripts import native_producer_validate_visualize as validator


_ENDPOINT_SHA = (
    "051f0d9e489355d312ad656532acf3ed8a2719774d8ef9843278818f5bd15adc"
)
_REASON = "raw_head_decoder_postprocess_contract_unresolved"
_BACKENDS = ("deepx_to_trt", "hailo10h_to_trt", "hailo8_to_trt")


def _raw_head_attestation() -> dict[str, Any]:
    return {
        "attested": True,
        "status": "passed",
        "stage": "raw_head",
        "endpoint": "raw_head",
        "reason": "raw_detection_tensor_structure_verified",
        "endpoint_contract_hash": _ENDPOINT_SHA,
        "tensor_signature": {
            "tensor_count": 3,
            "tensors": [
                {
                    "index": 0,
                    "name": "output",
                    "rank": 5,
                    "shape": [1, 3, 80, 80, 85],
                    "dtype": "float32",
                },
                {
                    "index": 1,
                    "name": "clone_1",
                    "rank": 5,
                    "shape": [1, 3, 40, 40, 85],
                    "dtype": "float32",
                },
                {
                    "index": 2,
                    "name": "clone_2",
                    "rank": 5,
                    "shape": [1, 3, 20, 20, 85],
                    "dtype": "float32",
                },
            ],
        },
    }


def _screening_row(backend: str = "deepx_to_trt") -> dict[str, Any]:
    policy = AccuracyGatePolicy().as_dict()
    policy_sha = validator._canonical_json_sha256(policy)
    row: dict[str, Any] = {
        # Real YOLOv7 b044 field constellation from the 20260727_082223 pack.
        "backend": backend,
        "model": "yolov7_paper",
        "case": "b044",
        "precision": (
            "uint8_dequant_fp16"
            if backend == "hailo8_to_trt"
            else "float32_layout_fp16"
        ),
        "setup_id": {
            "deepx_to_trt": "orin_nx_deepx_m1_01",
            "hailo10h_to_trt": "orin_nx_hailo10_01",
            "hailo8_to_trt": "orin_nx_hailo8_01",
        }[backend],
        "comparison_backend": {
            "deepx_to_trt": "deepx",
            "hailo10h_to_trt": "hailo10h",
            "hailo8_to_trt": "hailo8",
        }[backend],
        "task": "detection",
        "stage": "raw_head",
        "contract_family": "raw_head",
        "accelerator_output_stage": "raw_head",
        "accelerator_output_contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": _ENDPOINT_SHA,
        "accelerator_endpoint_contract_hash": _ENDPOINT_SHA,
        "output_endpoint_attestation": _raw_head_attestation(),
        "accelerator_output_endpoint_attestation": _raw_head_attestation(),
        "output_manifest_sha256": (
            "94c11021c9959cac35ab3c9ec68e714944e7645897877fab390585fbd33a3bda"
        ),
        "native_command_contract_sha256": (
            "09c4e8d55c2185bb0eaf430a95ea8fb19886323b6660ce047bd1fa090de8ecda"
        ),
        "native_split_quality_binding_sha256": (
            "1e16ab457d05b809ee479a948785af6777649f50ef3fcdeeebc576d57d11429d"
        ),
        "native_command_contract": {"fixture": "producer-sealed-command"},
        "native_split_quality_binding": {"fixture": "producer-sealed-binding"},
        "native_split_quality_required": True,
        "native_split_quality_binding_required": True,
        "native_split_quality_consumer_status": (
            "exact_quality_native_engine_command_and_boundary_match"
        ),
        "quality_first_binding_status": (
            "central_native_exact_engine_command_boundary_match"
        ),
        "quality_first_binding_errors": [],
        "central_quality_binding_status": "exact_identity_match",
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "native_row_ok": True,
        "report_ok": True,
        "buildable": True,
        "runtime_executable": True,
        "execution_validation_status": "passed",
        "tensor_ok": True,
        "strict_tensor_ok": True,
        "semantic_available": True,
        "semantic_ok": True,
        "semantic_validation_status": "passed",
        "self_reference_available": True,
        "self_reference_ok": True,
        "numerical_similarity_pass": True,
        "numerical_similarity_status": "passed",
        "structural_contract_pass": False,
        "contract_consistent": False,
        "structural_contract_reason": _REASON,
        "contract_gate_reason": _REASON,
        "claim_structural_gate_pass": False,
        "claim_structural_gate_reason": _REASON,
        "claim_ok_structural_clamped": True,
        "accuracy_gate_policy": policy,
        "accuracy_gate_policy_sha256": policy_sha,
        "task_quality_policy_sha256": policy_sha,
        "runtime_quality_gate_policy_sha256": policy_sha,
        "accuracy_gate_policy_match": True,
        "accuracy_gate_tier": "screening",
        "accuracy_gate_decision": "pass",
        "accuracy_gate_pass": True,
        "task_quality_gate": {
            "tier": "screening",
            "decision": "pass",
            "status": "pass",
            "policy": policy,
            "policy_sha256": policy_sha,
        },
        "task_quality_status": "pass",
        "task_quality_pass": True,
        "task_valid": True,
        "quality_valid": True,
        "evidence_complete": True,
        "host_postprocessing_evidence_status": "unavailable",
        "host_postprocessing_evidence_source": "none",
        "host_postprocessing_legacy_alias_conflict": False,
        "completed_task_endpoint_attested": None,
        "completed_task_endpoint_attestation_status": "",
        "completed_task_endpoint_attestation": {},
        "completed_task_endpoint_contract": None,
        "completed_task_stage": "",
        "completed_task_contract_family": "",
        "completed_task_endpoint_contract_hash": "",
        "completed_task_output_endpoint_id": "",
        "status": "structural_contract_failed",
        "ok": False,
        "claim_ok": False,
        "claim_ok_source": True,
        "eligible_for_ranking": False,
        # This legacy alias is intentionally not trusted as a claim grant.
        "e2e_claim_eligible": True,
    }
    if backend in {"hailo10h_to_trt", "hailo8_to_trt"}:
        row.update({
            "interface_contract_pass": True,
            "interface_check_pass": True,
            "interface_contract_status": (
                "verified_native_command_metadata_boundary_and_bridge"
            ),
        })
    return row


def _portable_binding_pass(**_: Any) -> tuple[dict[str, bool], str]:
    return (
        {"portable_engine_command_boundary_output_binding": True},
        "portable_binding_command_and_consumer_attestation_exact_match",
    )


def _set_quality_decision(
    row: dict[str, Any], decision: str,
) -> dict[str, Any]:
    row["accuracy_gate_decision"] = decision
    row["task_quality_status"] = decision
    row["task_quality_gate"]["decision"] = decision
    row["task_quality_gate"]["status"] = decision
    if decision == "pass":
        row.update({
            "task_quality_pass": True,
            "task_valid": True,
            "accuracy_gate_pass": True,
            "quality_valid": True,
            "evidence_complete": True,
            "precision_quality_verified": True,
        })
    elif decision == "fail":
        row.update({
            "task_quality_pass": False,
            "task_valid": False,
            "accuracy_gate_pass": False,
            "quality_valid": False,
            "evidence_complete": True,
            "precision_quality_verified": False,
        })
    elif decision == "inconclusive":
        row.update({
            "task_quality_pass": "inconclusive",
            "task_valid": "inconclusive",
            "accuracy_gate_pass": False,
            "quality_valid": False,
            "evidence_complete": False,
            "precision_quality_verified": False,
        })
    else:
        raise AssertionError(f"unsupported fixture decision: {decision}")
    return row


@pytest.mark.parametrize("backend", _BACKENDS)
def test_verified_raw_head_split_screening_gap_is_not_technical(
    monkeypatch: pytest.MonkeyPatch, backend: str,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row(backend)
    claim_axes_before = {
        field: row[field]
        for field in (
            "claim_ok", "eligible_for_ranking",
        )
    }

    assert validator._technical_quality_error(row) is False
    assert {
        field: row[field] for field in claim_axes_before
    } == claim_axes_before
    assert all(value is False for value in claim_axes_before.values())
    assert all(
        row.get(field) is not True
        for field in (
            "ranking_eligible", "performance_eligible", "energy_eligible",
            "pareto_eligible", "thesis_valid",
        )
    )


def test_negative_but_complete_semantic_decision_is_not_technical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row()
    row.update({
        "semantic_ok": False,
        "semantic_validation_status": "failed",
        "self_reference_ok": False,
        "numerical_similarity_pass": False,
        "numerical_similarity_status": "failed",
    })

    assert validator._technical_quality_error(row) is False
    assert row["claim_ok"] is False
    assert row["eligible_for_ranking"] is False
    assert row.get("energy_eligible") is not True


@pytest.mark.parametrize(
    ("backend", "decision", "expected_status"),
    [
        ("deepx_to_trt", "inconclusive", "diagnostic_technical_pass"),
        ("hailo10h_to_trt", "pass", "diagnostic_technical_pass"),
        (
            "hailo8_to_trt",
            "fail",
            "diagnostic_metric_threshold_warning",
        ),
    ],
)
def test_smoke_policy_keeps_complete_raw_head_screening_non_technical(
    monkeypatch: pytest.MonkeyPatch,
    backend: str,
    decision: str,
    expected_status: str,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _set_quality_decision(_screening_row(backend), decision)

    validator._apply_smoke_diagnostic_policy([row])

    assert row["status"] == expected_status
    assert row["ok"] is True
    assert row["diagnostic_only"] is True
    assert row["structural_contract_pass"] is False
    assert row["claim_ok"] is False
    assert row["eligible_for_ranking"] is False
    assert row["performance_claim_eligible"] is False
    assert row["energy_claim_eligible"] is False
    assert validator._technical_quality_error(row) is False


@pytest.mark.parametrize(
    ("field", "tampered"),
    [
        ("native_row_ok", False),
        ("report_ok", False),
        ("buildable", False),
        ("runtime_executable", False),
        ("tensor_ok", False),
        ("strict_tensor_ok", False),
        ("endpoint_contract_complete", False),
        ("endpoint_contract_hash", "0" * 64),
        ("semantic_available", False),
        ("semantic_ok", None),
        ("numerical_similarity_pass", False),
        ("self_reference_available", False),
        ("accuracy_gate_decision", "unavailable"),
        ("accuracy_gate_policy_match", False),
        ("central_quality_evidence_verified", False),
        ("precision_quality_verified", False),
        ("quality_first_binding_status", "binding_missing"),
        ("native_split_quality_consumer_status", "binding_failed"),
        ("claim_ok", True),
        ("eligible_for_ranking", True),
        ("energy_eligible", True),
    ],
)
def test_missing_or_manipulated_screening_evidence_remains_technical(
    monkeypatch: pytest.MonkeyPatch, field: str, tampered: Any,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row()
    row[field] = tampered

    assert validator._technical_quality_error(row) is True


def test_deepx_performance_input_mode_survives_projection(
    tmp_path: Path,
) -> None:
    row = {
        "backend": "native_full_deepx",
        "case": "full",
        "performance_input_contract_mode": "explicit",
    }
    native_report = {"performance_input_contract_mode": "explicit"}
    dump_payload = {"input_contract_mode": "explicit"}
    mode, status = validator._project_performance_input_contract_mode(
        row, native_report, dump_payload,
    )
    row["performance_input_contract_mode"] = mode
    manifest = tmp_path / "dump.json"
    manifest.write_text(json.dumps(dump_payload), encoding="utf-8")

    assert status == "projected_consistent"
    assert mode == "explicit"
    gate = validator._native_full_e2e_contract_gate(
        manifest, row, "classification",
    )
    assert gate["e2e_claim_eligible"] is True
    assert gate["e2e_contract_reason"] == (
        "native_full_non_detection_explicit_input_contract"
    )


def test_deepx_performance_input_mode_conflict_fails_closed(
    tmp_path: Path,
) -> None:
    row = {
        "backend": "native_full_deepx",
        "case": "full",
        "performance_input_contract_mode": "explicit",
    }
    dump_payload = {"input_contract_mode": "explicit"}
    mode, status = validator._project_performance_input_contract_mode(
        row,
        {"performance_input_contract_mode": "autodetect"},
        dump_payload,
    )
    row["performance_input_contract_mode"] = mode
    manifest = tmp_path / "dump.json"
    manifest.write_text(json.dumps(dump_payload), encoding="utf-8")

    assert status == "conflict"
    gate = validator._native_full_e2e_contract_gate(
        manifest, row, "classification",
    )
    assert gate["e2e_claim_eligible"] is False
    assert gate["e2e_contract_reason"] == (
        "deepx_performance_input_contract_not_explicit"
    )


def test_hailo_boundary_verifier_failure_remains_technical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row("hailo8_to_trt")
    row["interface_contract_pass"] = False

    assert validator._technical_quality_error(row) is True


def test_portable_engine_command_boundary_verifier_failure_is_technical(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validator,
        "bind_quality_to_native_split",
        lambda **_: (None, "native_split_quality_consumer_attestation_sha256_mismatch"),
    )

    assert validator._technical_quality_error(_screening_row()) is True


@pytest.mark.parametrize(
    "mutation",
    ("native_full_backend", "native_full_case", "different_reason"),
)
def test_non_split_or_other_structural_failures_remain_technical(
    monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row()
    if mutation == "native_full_backend":
        row["backend"] = "native_full_deepx"
    elif mutation == "native_full_case":
        row["case"] = "full"
    else:
        row["structural_contract_reason"] = "interface_contract_failed"

    assert validator._technical_quality_error(row) is True


@pytest.mark.parametrize(
    "mutation",
    ("top_level_final", "quality_gate_final", "coherent_final_policy"),
)
def test_final_policy_variants_remain_technical(
    monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row()
    if mutation == "top_level_final":
        row["accuracy_gate_tier"] = "final"
    elif mutation == "quality_gate_final":
        row["task_quality_gate"]["tier"] = "final"
    else:
        policy = copy.deepcopy(row["accuracy_gate_policy"])
        policy["dataset_tier"] = "final"
        policy["frozen_before_final_campaign"] = True
        policy_sha = validator._canonical_json_sha256(policy)
        row.update({
            "accuracy_gate_policy": policy,
            "accuracy_gate_policy_sha256": policy_sha,
            "task_quality_policy_sha256": policy_sha,
            "runtime_quality_gate_policy_sha256": policy_sha,
            "accuracy_gate_tier": "final",
        })
        row["task_quality_gate"].update({
            "tier": "final",
            "policy": policy,
            "policy_sha256": policy_sha,
        })

    assert validator._technical_quality_error(row) is True


def test_populated_completed_tail_with_unresolved_reason_is_contradictory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        validator, "bind_quality_to_native_split", _portable_binding_pass,
    )
    row = _screening_row()
    row["completed_task_endpoint_attested"] = True
    row["completed_task_endpoint_attestation_status"] = "passed"

    assert validator._technical_quality_error(row) is True
