from __future__ import annotations

from collections import Counter
import importlib.util
import json
from pathlib import Path
import sys
from typing import Any

import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_planner():
    path = ROOT / "scripts" / "native_producer_energy_plan.py"
    spec = importlib.util.spec_from_file_location(
        "v2727_native_energy_runtime_admission", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_energy_runner():
    path = (
        ROOT / "scripts"
        / "run_native_producer_energy_from_summary.py"
    )
    spec = importlib.util.spec_from_file_location(
        "v2727_native_energy_runtime_executor", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _runtime_matrix_rows() -> list[dict[str, Any]]:
    models = ("resnet50", "yolo26s", "yolov7_paper")
    setups = (
        (
            "hailo8_to_trt",
            "native_full_hailo8",
            "orin_nx_hailo8_01",
            "hailo8",
        ),
        (
            "hailo10h_to_trt",
            "native_full_hailo10h",
            "orin_nx_hailo10_01",
            "hailo10h",
        ),
        (
            "deepx_to_trt",
            "native_full_deepx",
            "orin_nx_deepx_m1_01",
            "deepx",
        ),
    )
    rows: list[dict[str, Any]] = []
    for split_backend, full_backend, setup, comparison in setups:
        for model in models:
            task = "classification" if model == "resnet50" else "detection"
            for case in ("b001", "b002", "b003"):
                rows.append({
                    "ok": True,
                    "backend": split_backend,
                    "model": model,
                    "case": case,
                    "precision": "runtime_precision",
                    "setup_id": setup,
                    "comparison_backend": comparison,
                    "part2_input_count": 1,
                    "task": task,
                    "fps_makespan": 10.0,
                    "native_command_contract": {
                        "contract_sha256": "a" * 64,
                    },
                })
            for backend, fps in (
                (full_backend, 8.0),
                ("native_full_tensorrt", 12.0),
            ):
                rows.append({
                    "ok": True,
                    "backend": backend,
                    "model": model,
                    "case": "full",
                    "precision": "runtime_precision",
                    "setup_id": setup,
                    "comparison_backend": comparison,
                    "task": task,
                    "fps_makespan": fps,
                    "full_command_contract": {
                        "contract_sha256": "b" * 64,
                    },
                })
    assert len(rows) == 45
    return rows


def test_runtime_success_mode_plans_the_complete_45_row_native_matrix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    planner = _load_planner()
    rows = _runtime_matrix_rows()
    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    out = tmp_path / "energy-plan"
    summary.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": []}), encoding="utf-8")
    (tmp_path / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": 45,
            "present_expected_row_count": 45,
            "successful_expected_row_count": 45,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
            "row_presence_complete": True,
            "present_expected_rows": rows,
            "failed_expected_rows": [],
            "missing_expected_rows": [],
        }),
        encoding="utf-8",
    )

    def verified_contract(
        raw: Any, *, expected_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return {
            "contract_sha256": str(
                (raw or {}).get("contract_sha256") or "c" * 64
            ),
            "backend": expected_identity["backend"],
            "model": expected_identity["model"],
            "case": expected_identity["case"],
            "setup_id": expected_identity["setup_id"],
            "comparison_backend": expected_identity[
                "comparison_backend"
            ],
            "runtime_options": {},
            "energy_workload": {},
            "artifacts": {},
        }, "verified_test_contract"

    monkeypatch.setattr(
        planner, "_verify_full_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_energy_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_split_part2_input_contract",
        lambda _contract: ({"inputs": [{}]}, "verified_test_part2"),
    )
    monkeypatch.setattr(
        planner,
        "_split_quality_energy_evidence",
        lambda *_args, **_kwargs: ({
            "native_split_quality_required": False,
            "native_split_energy_binding_valid": True,
            "native_split_energy_binding_status":
                "runtime_measurement_quality_not_available",
        }, "runtime_measurement_quality_not_available"),
    )
    monkeypatch.setattr(
        planner,
        "split_energy_runtime_argv",
        lambda *_args, **_kwargs: ["python", "split-energy.py"],
    )
    monkeypatch.setattr(
        planner,
        "_full_runtime_argv",
        lambda *_args, **_kwargs: ["python", "full-energy.py"],
    )
    monkeypatch.setattr(
        planner,
        "_split_preflight_argv",
        lambda *_args, **_kwargs: ["python", "split-preflight.py"],
    )
    monkeypatch.setattr(
        planner,
        "_full_preflight_argv",
        lambda *_args, **_kwargs: ["python", "full-preflight.py"],
    )
    monkeypatch.setattr(
        planner,
        "_process_local_runtime_environment",
        lambda _contract: {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(planner.__file__),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo8-ssh", "hailo8-host",
            "--hailo10-ssh", "hailo10-host",
            "--deepx-ssh", "deepx-host",
            "--duration-s", "1",
            "--screening-energy",
            "--measure-all-runtime-successful",
        ],
    )

    assert planner.main() == 0
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )
    assert payload["measure_all_runtime_successful"] is True
    assert payload["energy_matrix_expected_count"] == 45
    assert payload["energy_plan_included_count"] == 45
    assert payload["energy_plan_excluded_count"] == 0
    assert payload["energy_plan_coverage_contract_valid"] is True
    assert payload["preflight_status"] == "passed"
    assert len(payload["rows"]) == 45
    assert payload["excluded_rows"] == []
    assert payload["pairing_policy"] == (
        "measurement_independent_quality_pairing_posthoc"
    )
    assert Counter(
        row["setup_id"] for row in payload["rows"]
    ) == {
        "orin_nx_hailo8_01": 15,
        "orin_nx_hailo10_01": 15,
        "orin_nx_deepx_m1_01": 15,
    }
    assert all(
        row["energy_quality_admission"]["admission_scope"]
        == "native_runtime_observation"
        and row["diagnostic_only"] is True
        and row["claim_eligible"] is False
        and row["energy_claim_eligible"] is False
        for row in payload["rows"]
    )
    runner = _load_energy_runner()
    assert all(
        runner._prepare_measurement_execution(
            row,
            payload,
            allowed_root=tmp_path,
            validate_only=True,
        )["validated"]
        is True
        for row in payload["rows"]
    )


def test_runtime_observation_admission_never_becomes_a_claim() -> None:
    from onnx_splitpoint_tool.native_energy_quality_admission import (
        canonical_json_sha256,
        verify_sealed_energy_quality_admission,
    )

    admission = {
        "schema": "onnx-splitpoint/native-energy-quality-admission",
        "schema_version": 1,
        "admission_scope": "native_runtime_observation",
        "runtime_observation_reason": (
            "quality_binding_unavailable"
        ),
        "backend": "deepx_to_trt",
        "model": "yolo26s",
        "case": "b026",
        "setup_id": "orin_nx_deepx_m1_01",
        "precision": "float32_layout_fp16",
        "comparison_backend": "deepx",
        "successful_command_contract_sha256": "c" * 64,
        "central_quality_evidence_verified": False,
        "precision_quality_binding_verified": False,
        "task_quality_observation_valid": False,
        "accuracy_gate_pass": False,
        "quality_provenance_complete": False,
        "quality_claim_result_verified": False,
        "diagnostic_only": True,
        "screening_comparable": False,
        "claim_comparable": False,
        "energy_claim_eligible": False,
    }
    admission["admission_sha256"] = canonical_json_sha256(admission)
    row = {
        **{
            field: admission[field]
            for field in (
                "backend",
                "model",
                "case",
                "setup_id",
                "precision",
                "comparison_backend",
                "successful_command_contract_sha256",
                "central_quality_evidence_verified",
                "precision_quality_binding_verified",
                "task_quality_observation_valid",
                "accuracy_gate_pass",
                "quality_provenance_complete",
                "quality_claim_result_verified",
                "diagnostic_only",
                "screening_comparable",
                "claim_comparable",
                "energy_claim_eligible",
            )
        },
        "energy_quality_admission": admission,
        "energy_quality_admission_sha256": admission[
            "admission_sha256"
        ],
        "claim_ok": False,
        "semantic_claim_ok": False,
        "claim_eligible": False,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_claim": False,
    }

    digest, status = verify_sealed_energy_quality_admission(
        row, required=True,
    )

    assert digest == admission["admission_sha256"]
    assert status == "sealed_energy_quality_admission_verified"


def test_executor_accepts_only_sealed_runtime_observation_binding() -> None:
    runner = _load_energy_runner()
    command_sha = "d" * 64
    evidence = {
        "schema": (
            "onnx-splitpoint/"
            "native-split-energy-runtime-observation"
        ),
        "schema_version": 1,
        "native_split_quality_required": True,
        "native_split_energy_binding_valid": False,
        "native_split_energy_binding_status":
            "quality_binding_unavailable",
        "native_command_contract_sha256": command_sha,
    }
    evidence["evidence_sha256"] = runner._canonical_json_sha256(
        evidence
    )
    row = {
        "backend": "deepx_to_trt",
        "native_split_quality_required": True,
        "native_split_energy_binding_valid": False,
        "native_split_energy_binding_status":
            "quality_binding_unavailable",
        "successful_command_contract_sha256": command_sha,
        "native_split_energy_quality_binding": evidence,
        "native_split_energy_quality_binding_sha256": evidence[
            "evidence_sha256"
        ],
    }

    observed_sha, status = (
        runner._verify_split_energy_quality_binding(
            row,
            admission_scope="native_runtime_observation",
        )
    )

    assert observed_sha == command_sha
    assert status == (
        "sealed_runtime_observation_binding_verified"
    )

    tampered = {
        **row,
        "native_split_energy_binding_status": "tampered",
    }
    with pytest.raises(
        ValueError,
        match="native_split_runtime_observation_binding_invalid",
    ):
        runner._verify_split_energy_quality_binding(
            tampered,
            admission_scope="native_runtime_observation",
        )


def test_verified_quality_remains_claimable_until_pairing_rejects_it() -> None:
    planner = _load_planner()
    digest = "e" * 64
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b039",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "precision": "float32_layout_fp16",
        "task": "classification",
    }
    validation = {
        **row,
        "semantic_ok": True,
        "contract_consistent": True,
        "top1_match": True,
        "claim_ok": True,
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_provenance_complete": True,
        "quality_claim_result_verified": True,
        "source_request_sha256": digest,
        "model_sha256": digest,
        "validation_dataset_sha256": digest,
        "validation_dataset_image_ids_sha256": digest,
        "validation_dataset_ground_truth_sha256": digest,
        "accuracy_gate_policy_sha256": digest,
        "task_quality_policy_sha256": digest,
        "runtime_quality_gate_policy_sha256": digest,
    }

    admission, status = (
        planner._prepair_energy_quality_admission(
            row,
            validation,
            setup=row["setup_id"],
            command_contract_sha256="f" * 64,
            screening_only=False,
            smoke_diagnostic=False,
            historical_diagnostic_only=False,
            runtime_observation_allowed=True,
        )
    )

    assert status == "shared_energy_quality_admission_verified"
    assert admission is not None
    assert admission["admission_scope"] == "native_energy"
    assert admission["diagnostic_only"] is False
    assert admission["claim_comparable"] is True
    assert admission["energy_claim_eligible"] is True

    downgraded = planner._native_runtime_observation_admission(
        admission,
        reason="pair_baseline_missing",
    )
    assert downgraded["admission_scope"] == (
        "native_runtime_observation"
    )
    assert downgraded["diagnostic_only"] is True
    assert downgraded["screening_comparable"] is False
    assert downgraded["claim_comparable"] is False
    assert downgraded["energy_claim_eligible"] is False


@pytest.mark.parametrize(
    "incomplete_expected_matrix",
    (False, True),
    ids=("complete_matrix", "incomplete_matrix"),
)
def test_negative_full_keeps_triplet_measured_but_blocks_pair_claim(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    incomplete_expected_matrix: bool,
) -> None:
    planner = _load_planner()
    matrix = _runtime_matrix_rows()
    rows = [matrix[index] for index in (0, 3, 4)]
    endpoint_sha = "d" * 64
    digest = "e" * 64
    for row in rows:
        row.update({
            "part2_input_count": (
                1
                if row["backend"] == "hailo8_to_trt"
                else None
            ),
            "task": "classification",
            "stage": "classification_logits",
            "contract_family": "classification_logits",
            "endpoint_contract_complete": True,
            "endpoint_contract_hash": endpoint_sha,
            "output_endpoint_id": (
                "classification:classification_logits:"
                f"{endpoint_sha}"
            ),
            "model_sha256": digest,
        })

    summary = tmp_path / "native_producer_summary.json"
    validation = tmp_path / "native_producer_validation_summary.json"
    out = tmp_path / "energy-plan"
    summary.write_text(json.dumps({"rows": rows}), encoding="utf-8")
    validation.write_text(json.dumps({"rows": [
        {
            **row,
            "ok": True,
            "error": "",
            "task": "classification",
            "top1_match": True,
            "semantic_ok": True,
            "contract_consistent": True,
            "claim_ok": (
                row["backend"] != "native_full_hailo8"
            ),
            "central_quality_evidence_verified": True,
            "precision_quality_verified": True,
            "precision_quality_binding_verified": True,
            "task_quality_observation_valid": True,
            "accuracy_gate_pass": (
                row["backend"] != "native_full_hailo8"
            ),
            "quality_claim_result_verified": True,
            "runtime_precision_identity": "runtime_precision",
            "source_request_sha256": digest,
            "validation_dataset_sha256": digest,
            "validation_dataset_image_ids_sha256": digest,
            "validation_dataset_ground_truth_sha256": digest,
            "accuracy_gate_policy_sha256": digest,
            "task_quality_policy_sha256": digest,
            "runtime_quality_gate_policy_sha256": digest,
        }
        for row in rows
    ]}), encoding="utf-8")
    missing_rows = [matrix[1]] if incomplete_expected_matrix else []
    (tmp_path / "native_expected_matrix.json").write_text(
        json.dumps({
            "expected_row_count": 3 + len(missing_rows),
            "present_expected_row_count": 3,
            "successful_expected_row_count": 3,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": len(missing_rows),
            "row_presence_complete": not incomplete_expected_matrix,
            "present_expected_rows": rows,
            "failed_expected_rows": [],
            "missing_expected_rows": missing_rows,
        }),
        encoding="utf-8",
    )

    authority = {
        "valid": True,
        "mode": "required",
        "native_split_quality_required": True,
        "workflow_version": "test",
        "run_id": "test",
    }
    monkeypatch.setattr(
        planner,
        "_energy_split_quality_authority",
        lambda _path: dict(authority),
    )

    def apply_authority(
        row: dict[str, Any],
        _authority: dict[str, Any],
    ) -> dict[str, Any]:
        row.update({
            "native_split_quality_required": True,
            "native_split_quality_binding_required": True,
            "native_split_quality_authority_valid": True,
            "native_split_quality_authority_mode": "required",
        })
        return row

    def verified_contract(
        raw: Any, *, expected_identity: dict[str, Any],
    ) -> tuple[dict[str, Any], str]:
        return {
            "contract_sha256": str(
                (raw or {}).get("contract_sha256") or "f" * 64
            ),
            **{
                field: expected_identity[field]
                for field in (
                    "backend", "model", "case", "setup_id",
                    "comparison_backend",
                )
            },
            "runtime_options": {},
            "energy_workload": {},
            "artifacts": {},
        }, "verified_test_contract"

    monkeypatch.setattr(
        planner, "apply_native_split_quality_authority",
        apply_authority,
    )
    monkeypatch.setattr(
        planner, "_verify_full_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_energy_command_contract", verified_contract,
    )
    monkeypatch.setattr(
        planner, "verify_native_split_part2_input_contract",
        lambda _contract: ({"inputs": [{}]}, "verified_test_part2"),
    )
    monkeypatch.setattr(
        planner,
        "_split_quality_energy_evidence",
        lambda *_args, **_kwargs: ({
            "native_split_quality_required": True,
            "native_split_energy_binding_valid": True,
            "native_split_energy_binding_status":
                "exact_test_binding_verified",
        }, "exact_test_binding_verified"),
    )
    monkeypatch.setattr(
        planner,
        "split_energy_runtime_argv",
        lambda *_args, **_kwargs: ["python", "split-energy.py"],
    )
    monkeypatch.setattr(
        planner,
        "_full_runtime_argv",
        lambda *_args, **_kwargs: ["python", "full-energy.py"],
    )
    monkeypatch.setattr(
        planner,
        "_split_preflight_argv",
        lambda *_args, **_kwargs: ["python", "split-preflight.py"],
    )
    monkeypatch.setattr(
        planner,
        "_full_preflight_argv",
        lambda *_args, **_kwargs: ["python", "full-preflight.py"],
    )
    monkeypatch.setattr(
        planner,
        "_process_local_runtime_environment",
        lambda _contract: {},
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(planner.__file__),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo8-ssh", "hailo8-host",
            "--duration-s", "1",
            "--measure-all-runtime-successful",
            "--final-all-split-energy",
        ],
    )

    assert planner.main() == 0
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )
    assert payload["energy_plan_included_count"] == 3
    assert payload["energy_plan_excluded_count"] == len(missing_rows)
    assert payload["energy_plan_coverage_contract_valid"] is (
        not incomplete_expected_matrix
    )
    assert payload["preflight"]["measurement_start_allowed"] is True
    assert len(payload["rows"]) == 3
    assert payload["paired_groups"][0][
        "quality_pair_claim_eligible"
    ] is False
    assert all(
        row["energy_quality_admission"]["admission_scope"]
        == "native_runtime_observation"
        and row["diagnostic_only"] is True
        and row["screening_comparable"] is False
        and row["energy_quality_admission"][
            "screening_comparable"
        ] is False
        and row["claim_eligible"] is False
        and row["energy_claim_eligible"] is False
        for row in payload["rows"]
    )
    from onnx_splitpoint_tool.native_energy_quality_admission import (
        verify_sealed_energy_quality_admission,
    )

    for row in payload["rows"]:
        digest, status = verify_sealed_energy_quality_admission(
            row, required=True,
        )
        assert digest == row["energy_quality_admission_sha256"]
        assert status == "sealed_energy_quality_admission_verified"
