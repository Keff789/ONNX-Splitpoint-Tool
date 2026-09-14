from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Any

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
)
from onnx_splitpoint_tool.native_split_quality_authority import (
    SELECTION_FINGERPRINT_WORKFLOW,
)


ROOT = Path(__file__).resolve().parents[1]
REQUESTED_SELECTION_SHA256 = "1" * 64
RESOLVED_SELECTION_SHA256 = "2" * 64
SELECTION_FINGERPRINT = "c" * 64


def _load_fixture(name: str) -> ModuleType:
    path = ROOT / "tests" / name
    spec = importlib.util.spec_from_file_location(
        f"v270d_energy_authority_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_SMOKE_FIXTURE = _load_fixture("test_v270a_smoke_energy_policy.py")


def _bind_split_energy_part2_duplicates(split: dict[str, Any]) -> None:
    """Bring the historical split fixture up to the current Part-2 schema."""
    contract = split["native_command_contract"]
    artifacts = contract["artifacts"]
    boundary = contract["boundary_contract"]
    boundary.update({
        "metadata_path": artifacts["native_trt_meta"]["path"],
        "metadata_sha256": artifacts["native_trt_meta"]["sha256"],
    })
    contract["engine"] = artifacts["engine"]["path"]
    contract["engine_sha256"] = artifacts["engine"]["sha256"]
    contract.pop("contract_sha256", None)
    contract_sha256 = canonical_json_sha256(contract)
    contract["contract_sha256"] = contract_sha256
    split["native_command_contract_sha256"] = contract_sha256

    attestation = split["native_split_quality_consumer_attestation"]
    attestation["command_contract_sha256"] = contract_sha256
    attestation.pop("attestation_sha256", None)
    attestation_sha256 = canonical_json_sha256(attestation)
    attestation["attestation_sha256"] = attestation_sha256
    split[
        "native_split_quality_consumer_attestation_sha256"
    ] = attestation_sha256


def _write_authority(
    run: Path, *, stage_fingerprint: str = SELECTION_FINGERPRINT,
) -> None:
    reports = run / "reports"
    reports.mkdir(parents=True)
    manifest = {
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1,
        "run_id": run.name,
        "workflow_version": SELECTION_FINGERPRINT_WORKFLOW,
        "current_workflow_version": SELECTION_FINGERPRINT_WORKFLOW,
        "current_tool_version": "2.70.6",
        "execution_sessions": [{
            "workflow_version": SELECTION_FINGERPRINT_WORKFLOW,
            "tool_version": "2.70.6",
        }],
        "profile_start_snapshot": {
            "snapshot_sha256": "a" * 64,
            "requested_selection": {
                "snapshot_sha256": REQUESTED_SELECTION_SHA256,
            },
            "resolved_selection": {
                "snapshot_sha256": RESOLVED_SELECTION_SHA256,
            },
            "selection_fingerprint": SELECTION_FINGERPRINT,
        },
    }
    stage = {
        "schema": "onnx-splitpoint/native-producer-stage",
        "schema_version": 3,
        "run_id": run.name,
        "workflow_version": SELECTION_FINGERPRINT_WORKFLOW,
        "tool_version": "2.70.6",
        "profile_start_snapshot_sha256": "a" * 64,
        "profile_selection_snapshot_sha256": REQUESTED_SELECTION_SHA256,
        "profile_selection_fingerprint": stage_fingerprint,
        "native_split_quality_first": {"required": True},
    }
    (run / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8",
    )
    (reports / "native_producer_stage.json").write_text(
        json.dumps(stage), encoding="utf-8",
    )


def _write_energy_inputs(run: Path) -> tuple[Path, Path]:
    reports = run / "reports"
    split = _SMOKE_FIXTURE._replayable_quality_bound_split(
        run / "split_evidence",
    )
    _bind_split_energy_part2_duplicates(split)
    input_sha256 = split["native_command_contract"]["input_image_sha256"]
    common_full: dict[str, Any] = {
        "ok": True,
        "model": "yolo26s",
        "case": "full",
        "precision": "fp16",
        "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8",
        "task": "classification",
        "_test_task": "classification",
        "_test_input_sha256": input_sha256,
    }
    vendor_full = _SMOKE_FIXTURE._with_runtime_contract({
        **common_full,
        "backend": "native_full_hailo8",
        "fps_makespan": 8.0,
    })
    tensorrt_full = _SMOKE_FIXTURE._with_runtime_contract({
        **common_full,
        "backend": "native_full_tensorrt",
        "fps_makespan": 12.0,
    })
    rows = [split, vendor_full, tensorrt_full]
    summary = reports / "native_producer_summary.json"
    summary.write_text(json.dumps({"rows": rows}), encoding="utf-8")

    identity_fields = (
        "backend", "model", "case", "precision", "setup_id",
        "comparison_backend",
    )
    split_validation = {
        **{field: split[field] for field in identity_fields},
        "ok": True,
        "error": "",
        "task": "classification",
        "diagnostic_only": True,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "contract_consistent": True,
        "semantic_ok": True,
        "top1_match": True,
        "accuracy_gate_decision": "fail",
        "accuracy_gate_pass": False,
        "quality_claim_result_verified": False,
        "claim_ok": False,
        "energy_claim_eligible": False,
        "source_request_sha256": split["source_request_sha256"],
        "model_sha256": str(split.get("model_sha256") or "6" * 64),
        "validation_dataset_sha256": str(
            split.get("validation_dataset_sha256") or "7" * 64
        ),
        "validation_dataset_image_ids_sha256": str(
            split.get("validation_dataset_image_ids_sha256")
            or "8" * 64
        ),
        "validation_dataset_ground_truth_sha256": str(
            split.get("validation_dataset_ground_truth_sha256")
            or "9" * 64
        ),
        "accuracy_gate_policy_sha256": str(
            split.get("accuracy_gate_policy_sha256") or "a" * 64
        ),
        "task_quality_policy_sha256": str(
            split.get("accuracy_gate_policy_sha256") or "a" * 64
        ),
        "runtime_quality_gate_policy_sha256": str(
            split.get("accuracy_gate_policy_sha256") or "a" * 64
        ),
        "native_split_quality_source_request_sha256": split[
            "native_split_quality_source_request_sha256"
        ],
        "native_split_quality_central_result_sha256": split[
            "native_split_quality_central_result_sha256"
        ],
        "native_split_quality_selection_sha256": split[
            "native_split_quality_selection_sha256"
        ],
    }
    full_validations = [{
        **{field: row[field] for field in identity_fields},
        "ok": True,
        "error": "",
        "task": "classification",
        "runtime_precision_identity": "fp16",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "c" * 64,
        "quality_contract_sha256": "d" * 64,
        "preprocessing_contract_sha256": "e" * 64,
        "source_request_sha256": "f" * 64,
        "model_sha256": "1" * 64,
        "validation_dataset_sha256": "2" * 64,
        "validation_dataset_image_ids_sha256": "3" * 64,
        "validation_dataset_ground_truth_sha256": "4" * 64,
        "accuracy_gate_policy_sha256": "5" * 64,
        "task_quality_policy_sha256": "5" * 64,
        "runtime_quality_gate_policy_sha256": "5" * 64,
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_claim_result_verified": True,
        "claim_ok": True,
        "semantic_ok": True,
        "top1_match": True,
        "contract_consistent": True,
    } for row in (vendor_full, tensorrt_full)]
    validation = reports / "native_producer_validation_summary.json"
    validation.write_text(
        json.dumps({"rows": [split_validation, *full_validations]}),
        encoding="utf-8",
    )
    return summary, validation


def _run_energy_plan(
    tmp_path: Path, *, stage_fingerprint: str = SELECTION_FINGERPRINT,
) -> dict[str, Any]:
    run = tmp_path / "eval-native-split-001"
    _write_authority(run, stage_fingerprint=stage_fingerprint)
    summary, validation = _write_energy_inputs(run)
    out = tmp_path / "energy_plan"
    completed = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "native_producer_energy_plan.py"),
            "--summary", str(summary),
            "--validation-summary", str(validation),
            "--out-dir", str(out),
            "--hailo8-ssh", "diagnostic-host",
            "--duration-s", "1",
            "--smoke-diagnostic",
        ],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    return json.loads(
        (out / "native_producer_energy_plan.json").read_text(
            encoding="utf-8",
        )
    )


def test_v270d_selection_fingerprint_admits_complete_energy_pair(
    tmp_path: Path,
) -> None:
    payload = _run_energy_plan(tmp_path)
    authority = payload["native_split_quality_authority"]
    assert authority["valid"] is True
    assert authority["mode"] == "required"
    assert authority["selection_identity_mode"] == (
        "canonical_selection_fingerprint"
    )
    assert authority[
        "profile_requested_selection_snapshot_sha256"
    ] == REQUESTED_SELECTION_SHA256
    assert authority[
        "profile_resolved_selection_snapshot_sha256"
    ] == RESOLVED_SELECTION_SHA256
    assert authority["profile_selection_fingerprint"] == SELECTION_FINGERPRINT

    assert payload["pair_count"] == 1
    assert payload["paired_missing_rows"] == []
    assert len(payload["rows"]) == 3
    split = next(
        row for row in payload["rows"]
        if row["backend"] == "hailo8_to_trt"
    )
    assert split["native_split_energy_binding_valid"] is True
    assert split["native_split_energy_binding_status"] == (
        "portable_join_and_semantic_payload_bytes_rehashed"
    )
    assert split["native_split_quality_authority_workflow_version"] == (
        SELECTION_FINGERPRINT_WORKFLOW
    )
    assert split["native_split_quality_authority_run_id"] == (
        "eval-native-split-001"
    )


def test_v270d_stage_fingerprint_mismatch_is_downstream_nonclaim_annotation(
    tmp_path: Path,
) -> None:
    payload = _run_energy_plan(tmp_path, stage_fingerprint="d" * 64)
    authority = payload["native_split_quality_authority"]
    assert authority["valid"] is False
    assert "native_stage_selection_fingerprint_mismatch" in authority["errors"]
    assert payload["pair_count"] == 1
    assert payload["paired_missing_rows"] == []
    assert payload["energy_plan_included_count"] == 3
    assert payload["energy_plan_excluded_count"] == 0
    assert payload["preflight"]["measurement_start_allowed"] is True
    assert payload["excluded_rows"] == []
    assert len(payload["rows"]) == 3

    split = next(
        row for row in payload["rows"]
        if row["backend"] == "hailo8_to_trt"
    )
    assert split["native_split_energy_binding_valid"] is False
    assert split["native_split_energy_binding_status"] == (
        "native_split_quality_authority_invalid"
    )
    assert split["diagnostic_only"] is True
    for claim_axis in (
        "claim_ok",
        "semantic_claim_ok",
        "claim_eligible",
        "energy_claim_eligible",
        "eligible_for_scientific_claim",
    ):
        assert split[claim_axis] is False
