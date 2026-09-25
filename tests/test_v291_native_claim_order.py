"""The CLI must bind central quality before finalizing its semantic claim."""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

from onnx_splitpoint_tool.accuracy_reporting import (
    DEFAULT_REPORTING_POLICY, assess_accuracy,
)
from onnx_splitpoint_tool.validation.accuracy_gates import AccuracyGatePolicy
from scripts import native_producer_validate_visualize as validator
from scripts import native_producer_energy_plan as energy_plan


@pytest.mark.parametrize("control", ["good", "semantic_failure", "missing_binding", "structural_failure"])
def test_validator_main_binds_before_final_claim(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, control: str,
) -> None:
    policy = AccuracyGatePolicy(
        dataset_tier="final", reporting_policy=copy.deepcopy(DEFAULT_REPORTING_POLICY),
    )
    # A compact custom producer exercises the ordinary exact central join.
    # The real H8/H10 artifact-bound replay is kept outside the source tree.
    row = dict(backend="fixture_native", model="resnet50", case="b001",
               setup_id="fixture_setup", comparison_backend="fixture", task="classification",
               precision="fp16", runtime_precision_identity="fp16", ok=True)
    provenance = dict(source_request_sha256="2" * 64, model_sha256="3" * 64,
                      validation_dataset_sha256="4" * 64,
                      validation_dataset_image_ids_sha256="5" * 64,
                      validation_dataset_ground_truth_sha256="6" * 64,
                      quality_contract_sha256="7" * 64,
                      preprocessing_contract_sha256="8" * 64,
                      policy_sha256=policy.sha256())
    identity = dict(schema_version=4, identity_valid=True, model_id="resnet50",
                    task="classification", case_id="b001", source_run_id="fixture_native",
                    setup_id="fixture_setup", variant="composed", endpoint_contract_hash="1" * 64,
                    runtime_precision_identity="fp16", **provenance)
    central = dict(model_id="resnet50", task="classification", case_id="b001",
                   source_run_id="fixture_native", source_setup_id="fixture_setup",
                   variant="composed", status="completed", technical_status="completed",
                   decision="reference_close", request_identity=identity,
                   endpoint_contract_hash="1" * 64, runtime_precision_identity="fp16",
                   primary=dict(metric="top1_accuracy", reference=0.8, candidate=0.8,
                                delta=0.0, ci_low=0.0, ci_high=0.0), n=5000,
                   reference_identity={"reference": "fixture"},
                   accuracy_assessment=assess_accuracy(0.8, 0.8, [0.0, 0.0]), **provenance)
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": [row]}))
    central_path = tmp_path / "central.json"
    central_path.write_text(json.dumps({"results": [] if control == "missing_binding" else [central]}))
    dump = tmp_path / "dump.json"
    dump.write_text(json.dumps(dict(stage="classification_logits", contract_family="classification_logits",
                                    endpoint_contract_complete=control != "structural_failure",
                                    endpoint_contract_hash="1" * 64)))
    report = tmp_path / "native.json"
    report.write_text(json.dumps(row))
    monkeypatch.setattr(validator, "_eval_root_from_summary", lambda _: tmp_path)
    monkeypatch.setattr(validator, "_native_split_authority", lambda _: {})
    monkeypatch.setattr(validator, "_find_report", lambda *_: report)
    monkeypatch.setattr(validator, "_find_dump", lambda *_: (dump, "fixture"))
    monkeypatch.setattr(validator, "_find_reference_report", lambda *_: None)
    monkeypatch.setattr(validator, "_validate_tensor_dump", lambda *_: {"ok": True, "output_count": 1})
    semantic_ok = control != "semantic_failure"
    monkeypatch.setattr(validator, "_validate_classification", lambda *_: dict(
        ok=semantic_ok, semantic_available=True, semantic_ok=semantic_ok,
        top1_match=semantic_ok, top5_overlap=1.0 if semantic_ok else 0.0,
    ))
    monkeypatch.setattr(validator, "_full_onnx_self_reference_classification", lambda *a, **k: {"available": False})
    output = tmp_path / "validation"
    monkeypatch.setattr(sys, "argv", ["validator", "--summary", str(summary), "--out-dir", str(output),
                                    "--central-quality-summary", str(central_path),
                                    "--quality-gate-json", json.dumps(policy.as_dict())])
    assert validator.main() == 0
    result = json.loads((output / "native_producer_validation_summary.json").read_text())["rows"][0]
    assert result["semantic_ok"] is semantic_ok
    assert result["central_quality_evidence_verified"] is (control not in {"missing_binding", "structural_failure"})
    assert result["quality_claim_result_verified"] is (control == "good")
    assert result["claim_ok"] is (control == "good")
    assert result["status"] == "claim_ok" if control == "good" else result["status"] != "claim_ok"
    admission, _ = energy_plan._prepair_energy_quality_admission(
        row, result, setup="fixture_setup", command_contract_sha256="9" * 64,
        screening_only=True, smoke_diagnostic=False, historical_diagnostic_only=False,
        runtime_observation_allowed=True,
    )
    assert admission is not None
    assert admission["quality_claim_result_verified"] is (control == "good")
    assert admission["diagnostic_only"] is True
    assert admission["energy_claim_eligible"] is False
