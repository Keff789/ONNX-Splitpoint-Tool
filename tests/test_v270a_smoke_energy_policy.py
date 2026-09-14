from __future__ import annotations

import copy
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from typing import Any, Mapping

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
    seal_native_command_contract,
)


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _load_test_fixture(name: str):
    path = ROOT / "tests" / name
    spec = importlib.util.spec_from_file_location(
        f"v270a_smoke_energy_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


_ENERGY_FIXTURE = _load_test_fixture("test_energy_final_contract_regressions.py")
_INTEGRITY_FIXTURE = _load_test_fixture(
    "test_v269f_native_split_final_energy_integrity.py"
)
_with_runtime_contract = _ENERGY_FIXTURE._with_runtime_contract
_authority_run = _INTEGRITY_FIXTURE._authority_run
_load_script = _INTEGRITY_FIXTURE._load_script
_quality_first_row = _INTEGRITY_FIXTURE._quality_first_row
_sha256 = _INTEGRITY_FIXTURE._sha256

_CLAIM_ELIGIBILITY_FIELDS = {
    "claim_ok",
    "semantic_claim_ok",
    "claim_eligible",
    "performance_claim_eligible",
    "energy_claim_eligible",
    "eligible_for_ranking",
    "eligible_for_energy_results_import",
    "eligible_for_scientific_claim",
    "energy_efficiency_claim_eligible",
    "scientific_primary_claim_eligible",
    "candidate_eligible_for_scientific_primary",
    "eligible_for_scientific_primary",
}


def _assert_claim_and_eligibility_fields_false(value: Any) -> None:
    """Reject a truthy or tri-state claim field anywhere in a Smoke artifact."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            if key in _CLAIM_ELIGIBILITY_FIELDS:
                assert item is False, f"Smoke field {key} must be explicit false, got {item!r}"
            _assert_claim_and_eligibility_fields_false(item)
    elif isinstance(value, list):
        for item in value:
            _assert_claim_and_eligibility_fields_false(item)


def _replayable_quality_bound_split(tmp_path: Path) -> dict[str, Any]:
    """Add only prepared-feed replay fields without weakening Quality binding."""

    final = _load_script("native_producer_final_report.py")
    row, output_manifest, _output_payload = _quality_first_row(tmp_path)
    command = copy.deepcopy(row["native_command_contract"])
    command.pop("contract_sha256", None)
    command["runtime_options"] = {
        **dict(command.get("runtime_options") or {}),
        "frames": 10,
        "warmup": 1,
        "queue_depth": 2,
        "dump_outputs": False,
        "dump_boundary": False,
        "copy_outputs": True,
        "build": False,
        "task": "detection",
        "hailo_format": "uint8",
        "energy_prepared_feed_capable": True,
        "prepared_input_bound": True,
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": 114,
        "letterbox_pad_value": 114,
        "device_id": "",
    }
    command["prepared_input_contract"] = {
        "format": "raw_rgb_uint8",
        "shape": [2, 2, 3],
        "dtype": "uint8",
        "layout": "HWC",
        "task": "detection",
        "preprocess_mode_requested": "auto",
        "preprocess_mode_effective": "letterbox",
        "letterbox_pad_value_requested": 114,
        "letterbox_pad_value_effective": 114,
        "letterbox_pad_value": 114,
        "pad_value_effective": 114,
        "source_image_sha256": str(command["input_image_sha256"]),
    }
    command["artifacts"] = dict(command["artifacts"])
    command["artifacts"]["prepared_input"] = {
        "path": "/opt/prepared_input.bin",
        "sha256": "9" * 64,
        "size_bytes": 12,
    }
    # P0.3's technical Part-2 proof cross-checks the command's duplicate
    # runtime paths against the sealed producer-local artifact proof.
    quality_artifacts = command[
        "native_split_quality_binding"
    ]["artifacts"]
    command["engine"] = quality_artifacts["engine"]["path"]
    command["engine_sha256"] = quality_artifacts["engine"]["sha256"]
    command["boundary_contract"] = {
        **dict(command.get("boundary_contract") or {}),
        "metadata_path": quality_artifacts["native_trt_meta"]["path"],
        "metadata_sha256": quality_artifacts["native_trt_meta"]["sha256"],
    }
    command = seal_native_command_contract(command)

    attestation = copy.deepcopy(row["native_split_quality_consumer_attestation"])
    attestation.pop("attestation_sha256", None)
    attestation["command_contract_sha256"] = command["contract_sha256"]
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    row.update({
        "native_command_contract": command,
        "native_command_contract_sha256": command["contract_sha256"],
        "native_split_quality_consumer_attestation": attestation,
        "ok": True,
        "fps_makespan": 11.0,
    })
    final_evidence = final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(row,),
    )
    assert final_evidence["native_split_final_portable_binding_valid"] is True
    assert final_evidence["native_split_semantic_binding_valid"] is True
    row.update(final_evidence)
    return row


def _smoke_energy_inputs(tmp_path: Path) -> tuple[Path, Path]:
    summary = _authority_run(
        tmp_path / "managed-smoke-run",
        workflow="v2.69f-hardware-smoke-native-energy-repair",
        required=True,
    )
    bound = _replayable_quality_bound_split(tmp_path / "bound-split")
    unbound = {
        "ok": True,
        "backend": "hailo8_to_trt",
        "model": "yolo26s",
        "case": "b099",
        "precision": "uint8_dequant_fp16",
        "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8",
        "task": "detection",
        "fps_makespan": 9.0,
    }
    vendor_full = _with_runtime_contract({
        "ok": True,
        "backend": "native_full_hailo8",
        "model": "yolo26s",
        "case": "full",
        "precision": "fp16",
        "execution_precision": "uint8",
        "full_runtime_precision": "uint8",
        "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8",
        "task": "detection",
        "_test_task": "detection",
        "fps_makespan": 8.0,
    })
    summary.write_text(
        json.dumps({"rows": [bound, unbound, vendor_full]}),
        encoding="utf-8",
    )

    identity_fields = (
        "backend", "model", "case", "precision", "setup_id",
        "comparison_backend",
    )
    validation_row = {
        **{field: bound[field] for field in identity_fields},
        "ok": True,
        "error": "",
        "task": "detection",
        "diagnostic_only": True,
        "central_quality_evidence_verified": True,
        "precision_quality_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "contract_consistent": True,
        "semantic_ok": True,
        "accuracy_gate_decision": "fail",
        "accuracy_gate_pass": False,
        "quality_claim_result_verified": False,
        "quality_gate_status": "metric_threshold_warning",
        "metric_threshold_miss_warning": True,
        # Deliberately emulate an upstream Standard-style positive semantic
        # decision.  The Smoke planner must still clamp every claim field.
        "claim_ok": True,
        "eligible_for_ranking": True,
        "energy_claim_eligible": False,
        "source_request_sha256": bound["source_request_sha256"],
        "model_sha256": str(bound.get("model_sha256") or "2" * 64),
        "validation_dataset_sha256": str(
            bound.get("validation_dataset_sha256") or "3" * 64
        ),
        "validation_dataset_image_ids_sha256": str(
            bound.get("validation_dataset_image_ids_sha256")
            or "4" * 64
        ),
        "validation_dataset_ground_truth_sha256": str(
            bound.get("validation_dataset_ground_truth_sha256")
            or "5" * 64
        ),
        "accuracy_gate_policy_sha256": str(
            bound.get("accuracy_gate_policy_sha256") or "6" * 64
        ),
        "task_quality_policy_sha256": str(
            bound.get("accuracy_gate_policy_sha256") or "6" * 64
        ),
        "runtime_quality_gate_policy_sha256": str(
            bound.get("accuracy_gate_policy_sha256") or "6" * 64
        ),
        "native_split_quality_source_request_sha256": bound[
            "native_split_quality_source_request_sha256"
        ],
        "native_split_quality_central_result_sha256": bound[
            "native_split_quality_central_result_sha256"
        ],
        "native_split_quality_selection_sha256": bound[
            "native_split_quality_selection_sha256"
        ],
    }
    validation = tmp_path / "native_producer_validation_summary.json"
    validation.write_text(
        json.dumps({"rows": [validation_row]}), encoding="utf-8",
    )
    return summary, validation


def _planner_command(
    summary: Path, validation: Path, out: Path,
) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts/native_producer_energy_plan.py"),
        "--summary", str(summary),
        "--validation-summary", str(validation),
        "--out-dir", str(out),
        "--hailo8-ssh", "diagnostic-host",
        "--duration-s", "1",
        "--allow-unpaired",
        "--smoke-diagnostic",
    ]


def test_real_smoke_energy_plan_uses_technical_membership_only(
    tmp_path: Path,
) -> None:
    summary, validation = _smoke_energy_inputs(tmp_path)
    out = tmp_path / "direct-plan"
    completed = subprocess.run(
        _planner_command(summary, validation, out),
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=30,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    payload = json.loads(
        (out / "native_producer_energy_plan.json").read_text(encoding="utf-8")
    )

    planned = {(row["backend"], row["case"]): row for row in payload["rows"]}
    assert set(planned) == {
        ("hailo8_to_trt", "b038"),
        ("native_full_hailo8", "full"),
    }
    split = planned[("hailo8_to_trt", "b038")]
    assert split["central_quality_evidence_verified"] is True
    assert split["accuracy_gate_pass"] is False
    assert split["quality_gate_status"] == "metric_threshold_warning"
    assert split["native_split_energy_binding_valid"] is True
    assert split["semantic_gate"] == "smoke_diagnostic_not_claimable"
    assert all(
        "--diagnostic-only" in row["measure_command"]
        and "--claim-exclusion-reason smoke_diagnostic_only"
        in row["measure_command"]
        for row in payload["rows"]
    )

    excluded = {
        (row.get("backend"), row.get("case")): row
        for row in payload["excluded_rows"]
    }
    assert excluded[("hailo8_to_trt", "b099")]["reason"] == (
        "successful_command_contract_missing_or_invalid"
    )
    assert ("native_full_hailo8", "full") not in excluded
    assert planned[("native_full_hailo8", "full")][
        "diagnostic_only"
    ] is True
    assert payload["smoke_diagnostic"] is True
    assert payload["diagnostic_only"] is True
    _assert_claim_and_eligibility_fields_false(payload)


def test_real_smoke_energy_runner_dry_run_preserves_diagnostic_only_policy(
    tmp_path: Path,
) -> None:
    summary, validation = _smoke_energy_inputs(tmp_path)
    out = tmp_path / "runner"
    command = [
        sys.executable,
        str(ROOT / "scripts/run_native_producer_energy_from_summary.py"),
        "--summary", str(summary),
        "--validation-summary", str(validation),
        "--out-dir", str(out),
        "--hailo8-ssh", "diagnostic-host",
        "--duration-s", "1",
        "--allow-unpaired",
        "--smoke-diagnostic",
        "--dry-run",
    ]
    completed = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        timeout=60,
        check=False,
    )
    assert completed.returncode == 0, (completed.stdout, completed.stderr)
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(encoding="utf-8")
    )
    assert report["status"] == "dry_run"
    assert report["smoke_diagnostic"] is True
    assert report["diagnostic_only"] is True
    assert report["started_measurement_count"] == 0
    assert {
        (item["row"]["backend"], item["row"]["case"])
        for item in report["rows"]
    } == {
        ("hailo8_to_trt", "b038"),
        ("native_full_hailo8", "full"),
    }
    assert all(item["dry_run"] is True for item in report["rows"])
    assert any(
        row.get("case") == "b099"
        and row.get("reason") == (
            "successful_command_contract_missing_or_invalid"
        )
        for row in report["plan_payload"]["excluded_rows"]
    )
    assert not any(
        row.get("backend") == "native_full_hailo8"
        for row in report["plan_payload"]["excluded_rows"]
    )
    _assert_claim_and_eligibility_fields_false(report)
