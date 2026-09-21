from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import pytest

from onnx_splitpoint_tool.native_command_contract import (
    canonical_json_sha256,
    seal_native_command_contract,
)
from onnx_splitpoint_tool.native_split_quality import (
    seal_native_split_quality_binding,
    select_central_native_split_quality_binding,
)


ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v269f_integrity_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _load_receipt_fixture():
    path = ROOT / "tests" / "test_v269f_native_split_receipt_validation.py"
    spec = importlib.util.spec_from_file_location(
        "v269f_integrity_receipt_fixture", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _semantic_manifest(root: Path, role: str) -> tuple[Path, Path]:
    payload_path = root / f"{role}.bin"
    payload_path.parent.mkdir(parents=True, exist_ok=True)
    payload_path.write_bytes((f"semantic-{role}-payload-v1").encode("ascii"))
    rows = [{
        "path": str(payload_path.resolve()),
        "sha256": _sha256(payload_path),
        "size_bytes": payload_path.stat().st_size,
    }]
    manifest = {
        "schema": "onnx-splitpoint/native-output-dump-manifest",
        "schema_version": 1,
        "task": "detection",
        "stage": "raw_head",
        "output_format": "raw_head",
        "contract_family": "raw_head",
        "endpoint_contract_complete": True,
        "endpoint_contract_hash": "a" * 64,
        "payload_artifacts": rows,
        "payload_artifacts_sha256": canonical_json_sha256(rows),
    }
    manifest_path = root / f"{role}_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
    )
    return manifest_path, payload_path


def _quality_first_row(tmp_path: Path) -> tuple[dict[str, Any], Path, Path]:
    fixture = _load_receipt_fixture()
    payload, _paths = fixture._fixture_payload(tmp_path / "producer")
    producer_binding = seal_native_split_quality_binding(payload)
    binding = select_central_native_split_quality_binding(
        producer_binding,
        source_request_sha256="3" * 64,
        central_result_sha256="8" * 64,
        central_identity={
            "eval_run_id": producer_binding["eval_run_id"],
            "source_run_id": producer_binding["source_run_id"],
            "model_id": "yolo26s", "task": "detection", "case_id": "b038",
            "setup_id": "hailo8_setup", "variant": "composed",
            "runtime_precision_identity": "uint8_dequant_fp16",
        },
    )
    row = fixture._native_row_for_binding(binding)
    output_manifest, output_payload = _semantic_manifest(
        tmp_path / "semantic", "output",
    )
    boundary_manifest, _boundary_payload = _semantic_manifest(
        tmp_path / "semantic", "boundary",
    )

    command = copy.deepcopy(row["native_command_contract"])
    command.pop("contract_sha256", None)
    selected_hashes = {
        "native_split_quality_source_request_sha256": binding[
            "source_request_sha256"
        ],
        "native_split_quality_central_result_sha256": binding[
            "central_result_sha256"
        ],
        "native_split_quality_selection_sha256": binding[
            "central_quality_selection_sha256"
        ],
    }
    command.update(selected_hashes)
    command["artifacts"]["semantic_output_manifest"] = {
        "path": str(output_manifest.resolve()),
        "sha256": _sha256(output_manifest),
        "size_bytes": output_manifest.stat().st_size,
    }
    command["artifacts"]["semantic_boundary_manifest"] = {
        "path": str(boundary_manifest.resolve()),
        "sha256": _sha256(boundary_manifest),
        "size_bytes": boundary_manifest.stat().st_size,
    }
    command = seal_native_command_contract(command)

    attestation = copy.deepcopy(row["native_split_quality_consumer_attestation"])
    attestation.pop("attestation_sha256", None)
    attestation.update(selected_hashes)
    attestation["command_contract_sha256"] = command["contract_sha256"]
    attestation["semantic_output_manifest_sha256"] = _sha256(output_manifest)
    attestation["semantic_boundary_manifest_sha256"] = _sha256(boundary_manifest)
    attestation["attestation_sha256"] = canonical_json_sha256(attestation)
    row.update({
        "native_split_quality_required": True,
        "native_split_quality_provenance_conflict": False,
        "eval_run_id": binding["eval_run_id"],
        "native_split_quality_binding": copy.deepcopy(binding),
        "native_split_quality_binding_sha256": binding["binding_sha256"],
        "native_command_contract": command,
        "native_command_contract_sha256": command["contract_sha256"],
        "native_split_quality_consumer_attestation": attestation,
        "native_split_quality_consumer_status": (
            "exact_quality_native_engine_command_and_boundary_match"
        ),
        **selected_hashes,
        "source_request_sha256": selected_hashes[
            "native_split_quality_source_request_sha256"
        ],
        "quality_evidence_verified": True,
        "performance_claim_eligible": True,
        "native_fifo_output_manifest": str(output_manifest),
        "native_fifo_boundary_manifest": str(boundary_manifest),
        "native_split_quality_authority": {
            "schema": "onnx-splitpoint/native-split-quality-authority",
            "schema_version": 1, "mode": "required", "valid": True,
            "native_split_quality_required": True,
            "workflow_version": "v2.69f-hardware-smoke-native-energy-repair",
            "run_id": "eval-native-split-001", "stage_run_id": "eval-native-split-001",
            "stage_required": True, "errors": [],
        },
    })
    return row, output_manifest, output_payload


def test_final_quality_marker_conflict_cannot_downgrade_to_legacy() -> None:
    final = _load_script("native_producer_final_report.py")
    fields = final._split_quality_fields(
        {
            "native_split_quality_binding_sha256": "1" * 64,
            "native_split_quality_consumer_status": "status-a",
        },
        {
            "native_split_quality_binding_sha256": "2" * 64,
            "native_split_quality_consumer_status": "status-b",
        },
    )
    assert fields["native_split_quality_required"] is True
    assert fields["native_split_quality_provenance_conflict"] is True
    assert "native_split_quality_binding_sha256" in fields[
        "native_split_quality_provenance_conflict_fields"
    ]


def test_final_rebinds_attestation_and_rehashes_semantic_payloads(
    tmp_path: Path,
) -> None:
    final = _load_script("native_producer_final_report.py")
    row, output_manifest, output_payload = _quality_first_row(tmp_path)

    verified = final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(row,),
    )
    assert verified["native_split_final_portable_binding_valid"] is True
    assert verified["native_split_semantic_binding_valid"] is True

    drifted = copy.deepcopy(row)
    drifted["native_split_quality_consumer_attestation"]["eval_run_id"] = (
        "different-eval"
    )
    rejected = final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(drifted,),
    )
    assert rejected["native_split_final_portable_binding_valid"] is False
    assert rejected["native_split_semantic_binding_valid"] is False

    output_payload.write_bytes(b"tampered-with-same-manifest")
    rejected = final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(row,),
    )
    assert rejected["native_split_semantic_binding_valid"] is False
    assert "payload_artifact" in rejected["native_split_semantic_binding_status"]


def test_energy_repeats_portable_join_and_payload_rehash_then_seals_plan_row(
    tmp_path: Path,
) -> None:
    final = _load_script("native_producer_final_report.py")
    energy = _load_script("native_producer_energy_plan.py")
    runner = _load_script("run_native_producer_energy_from_summary.py")
    row, output_manifest, output_payload = _quality_first_row(tmp_path)
    final_evidence = final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(row,),
    )
    row.update(final_evidence)
    evidence, status = energy._split_quality_energy_evidence(
        row, row["native_command_contract"],
    )
    assert status == "portable_join_and_semantic_payload_bytes_rehashed"
    assert evidence is not None
    assert evidence["native_split_energy_binding_valid"] is True

    plan_row = {
        "native_split_quality_required": True,
        "source_request_sha256": evidence[
            "native_split_quality_source_request_sha256"
        ],
        "successful_command_contract_sha256": evidence[
            "native_command_contract_sha256"
        ],
        "native_split_energy_quality_binding": copy.deepcopy(evidence),
        "native_split_energy_quality_binding_sha256": evidence["evidence_sha256"],
        "native_split_quality_consumer_attestation": copy.deepcopy(
            row["native_split_quality_consumer_attestation"]
        ),
    }
    for field in (
        "native_split_quality_binding_sha256",
        "native_split_quality_preselection_sha256",
        "native_split_quality_source_request_sha256",
        "native_split_quality_central_result_sha256",
        "native_split_quality_selection_sha256",
        "native_split_quality_eval_run_id",
        "native_split_quality_source_run_id",
        "native_split_quality_consumer_attestation_sha256",
        "native_split_quality_authority_workflow_version",
        "native_split_quality_authority_run_id",
        "native_split_semantic_output_manifest_sha256",
        "native_split_semantic_boundary_manifest_sha256",
    ):
        plan_row[field] = evidence[field]
    command_sha, seal_status = runner._verify_split_energy_quality_binding(plan_row)
    assert command_sha == evidence["native_command_contract_sha256"]
    assert seal_status == "sealed_quality_binding_verified"

    drifted = copy.deepcopy(plan_row)
    drifted["native_split_quality_preselection_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="preselection_sha256_drift"):
        runner._verify_split_energy_quality_binding(drifted)

    drifted = copy.deepcopy(plan_row)
    drifted["native_split_quality_central_result_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="central_result_sha256_drift"):
        runner._verify_split_energy_quality_binding(drifted)

    drifted = copy.deepcopy(plan_row)
    drifted["source_request_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="generic_source_request_sha256_drift"):
        runner._verify_split_energy_quality_binding(drifted)

    output_payload.write_bytes(b"energy-payload-drift")
    rejected, rejected_status = energy._split_quality_energy_evidence(
        row, row["native_command_contract"],
    )
    assert rejected is None
    assert "payload_artifact" in rejected_status


def test_energy_source_run_aliases_join_through_managed_allowlist(
    tmp_path: Path,
) -> None:
    final = _load_script("native_producer_final_report.py")
    energy = _load_script("native_producer_energy_plan.py")
    row, output_manifest, _output_payload = _quality_first_row(tmp_path)
    row.update(final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(row,),
    ))
    binding = row["native_split_quality_binding"]
    command = copy.deepcopy(row["native_command_contract"])
    command.pop("contract_sha256", None)
    command["native_split_quality_source_run_id"] = (
        "hailo8_to_trt"
    )
    command = seal_native_command_contract(command)
    row["native_command_contract"] = command
    row["native_command_contract_sha256"] = command["contract_sha256"]
    row["native_split_quality_source_run_id"] = "hailo8_to_tensorrt"
    attestation = copy.deepcopy(
        row["native_split_quality_consumer_attestation"]
    )
    attestation.pop("attestation_sha256", None)
    attestation["command_contract_sha256"] = command["contract_sha256"]
    attestation["attestation_sha256"] = canonical_json_sha256(
        attestation
    )
    row["native_split_quality_consumer_attestation"] = attestation

    evidence, status = energy._split_quality_energy_evidence(
        row, command,
    )

    assert status == (
        "portable_join_and_semantic_payload_bytes_rehashed"
    )
    assert evidence is not None
    assert evidence[
        "native_split_quality_source_run_id_canonical"
    ] == "hailo8_to_trt"
    assert evidence["native_split_quality_source_run_id_raw"] == {
        "binding": binding["source_run_id"],
        "native_result": "hailo8_to_tensorrt",
        "native_command": "hailo8_to_trt",
    }


@pytest.mark.parametrize(
    ("raw", "canonical"),
    [
        ("hailo10_to_trt", "hailo10h_to_trt"),
        ("hailo10h_to_tensorrt", "hailo10h_to_trt"),
        ("deepx_m1_to_tensorrt", "deepx_to_trt"),
        ("deepx_m1_to_trt", "deepx_to_trt"),
    ],
)
def test_energy_source_run_alias_helper_uses_shared_allowlist(
    raw: str, canonical: str,
) -> None:
    energy = _load_script("native_producer_energy_plan.py")

    assert energy._canonical_native_split_source_run_id(raw) == canonical


def test_energy_unknown_source_run_alias_fails_closed(
    tmp_path: Path,
) -> None:
    final = _load_script("native_producer_final_report.py")
    energy = _load_script("native_producer_energy_plan.py")
    row, output_manifest, _output_payload = _quality_first_row(tmp_path)
    row.update(final._verify_split_semantic_artifacts(
        result_path=None,
        manifest_path=output_manifest,
        manifest_sha256=_sha256(output_manifest),
        sources=(row,),
    ))
    row["native_split_quality_source_run_id"] = "unknown_accelerator"

    evidence, status = energy._split_quality_energy_evidence(
        row, row["native_command_contract"],
    )

    assert evidence is None
    assert status == (
        "native_split_quality_source_run_id_unknown_or_unsupported"
    )


@pytest.mark.parametrize(
    "payload",
    [
        '{"backend":"deepx_to_trt","backend":"deepx_to_trt"}',
        '{"backend":"deepx_to_trt","backend":"hailo8_to_trt"}',
        '{"outer":{"setup_id":"a","setup_id":"a"}}',
    ],
)
def test_energy_preflight_rejects_all_duplicate_json_keys(payload: str) -> None:
    preflight = _load_script("native_split_energy_preflight.py")
    with pytest.raises(ValueError, match="duplicate JSON object keys"):
        preflight._strict_json_text(payload, label="inline contract payload")


def test_energy_preflight_source_and_remote_mirror_use_technical_contract() -> None:
    source = (ROOT / "scripts/native_split_energy_preflight.py").read_text(
        encoding="utf-8",
    )
    mirror = (
        ROOT / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_split_energy_preflight.py"
    ).read_text(encoding="utf-8")
    assert source == mirror
    assert "verify_native_energy_command_contract" in source
    assert "verify_native_split_part2_input_contract" in source
    assert "downstream_annotation_not_preflighted" in source
    assert "object_pairs_hook" in source


@pytest.mark.parametrize(
    "filename",
    [
        "native_producer_combined_summary.json",
        "native_producer_validation_summary.json",
    ],
)
@pytest.mark.parametrize("second_value", ["request-a", "request-b"])
def test_energy_summary_inputs_reject_identical_and_conflicting_duplicate_keys(
    tmp_path: Path, filename: str, second_value: str,
) -> None:
    energy = _load_script("native_producer_energy_plan.py")
    path = tmp_path / filename
    path.write_text(
        '{"rows":[{' 
        '"native_split_quality_source_request_sha256":"request-a",'
        f'"native_split_quality_source_request_sha256":"{second_value}"'
        '}]}',
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        energy._load_rows(path)


@pytest.mark.parametrize("second_value", ["request-a", "request-b"])
def test_energy_measurement_runner_rejects_duplicate_keys_in_sealed_plan(
    tmp_path: Path, second_value: str,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    path = tmp_path / "native_producer_energy_plan.json"
    path.write_text(
        '{"rows":[{'
        '"native_split_quality_source_request_sha256":"request-a",'
        f'"native_split_quality_source_request_sha256":"{second_value}"'
        '}]}',
        encoding="utf-8",
    )
    assert runner._load_json(path) is None
    with pytest.raises(ValueError, match="duplicate JSON object key"):
        runner._strict_json_text(path.read_text(encoding="utf-8"), label=str(path))


@pytest.mark.parametrize(
    "mutation",
    [
        {"setup_id": ""},
        {"setup_id": "other-setup"},
        {"comparison_backend": ""},
        {"comparison_backend": "deepx"},
    ],
)
def test_current_quality_first_energy_validation_join_has_no_identity_fallback(
    mutation: dict[str, str],
) -> None:
    energy = _load_script("native_producer_energy_plan.py")
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b001",
        "precision": "fp16",
        "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8",
        "ok": True,
    }
    validation = {
        **row,
        "task": "classification",
        "top1_match": True,
        "claim_ok": True,
        "semantic_ok": True,
        "contract_consistent": True,
    }
    validations = {energy._validation_identity(validation): validation}
    accepted, reason, matched = energy._semantic_decision(
        row, validations, require_claim=True, exact_identity_required=True,
    )
    assert accepted is True
    assert matched is validation

    drifted = {**row, **mutation}
    accepted, reason, matched = energy._semantic_decision(
        drifted, validations, require_claim=True, exact_identity_required=True,
    )
    assert accepted is False
    assert reason == "native_validation_exact_identity_missing"
    assert matched is None


@pytest.mark.parametrize(
    ("target", "field", "value"),
    [
        ("row", "quality_evidence_verified", False),
        ("row", "performance_claim_eligible", False),
        ("validation", "setup_id", "other-setup"),
        ("validation", "comparison_backend", "deepx"),
        (
            "validation", "native_split_quality_source_request_sha256",
            "f" * 64,
        ),
    ],
)
def test_current_quality_first_energy_claim_join_is_exact_and_fail_closed(
    target: str, field: str, value: Any,
) -> None:
    energy = _load_script("native_producer_energy_plan.py")
    selected = {
        "native_split_quality_source_request_sha256": "1" * 64,
        "native_split_quality_central_result_sha256": "2" * 64,
        "native_split_quality_selection_sha256": "3" * 64,
        "source_request_sha256": "1" * 64,
    }
    row = {
        "backend": "hailo8_to_trt", "model": "resnet50", "case": "b001",
        "precision": "fp16", "setup_id": "hailo8_setup",
        "comparison_backend": "hailo8", "quality_evidence_verified": True,
        "performance_claim_eligible": True, **selected,
    }
    validation = {**row, "task": "classification"}
    verified, status = energy._current_quality_first_validation_join(
        row, validation,
    )
    assert verified is True
    assert status == "exact_final_validation_and_central_selection_join_verified"

    mutated_row = copy.deepcopy(row)
    mutated_validation = copy.deepcopy(validation)
    (mutated_row if target == "row" else mutated_validation)[field] = value
    verified, status = energy._current_quality_first_validation_join(
        mutated_row, mutated_validation,
    )
    assert verified is False
    assert status != "exact_final_validation_and_central_selection_join_verified"


def test_hailo8_manual_cli_without_quality_set_runs_diagnostic_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    matrix = _load_script("native_fifo_smoke_matrix.py")
    benchmark_set = tmp_path / "benchmark_set"
    (benchmark_set / "b001").mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(json.dumps({
        "task": "classification", "cases": [{"case_id": "b001"}],
    }), encoding="utf-8")
    image = benchmark_set / "resources/validation/classification/image.jpg"
    image.parent.mkdir(parents=True)
    image.write_bytes(b"diagnostic-image")
    from onnx_splitpoint_tool import native_progress
    from types import SimpleNamespace
    preparations = []
    def prepare(command, **kwargs):
        assert '--no-run' in command and '--no-build' not in command
        assert kwargs['timeout'] == 300
        preparations.append(command)
        return SimpleNamespace(returncode=0, elapsed_s=.01, stdout='fixture wrapper ready')
    monkeypatch.setattr(native_progress, 'run_streaming', prepare)

    def fake_run(command: list[str], timeout: float = 0.0, env=None):
        if "native_fifo_capability_report.py" in " ".join(command):
            report = benchmark_set / "analysis_tables/native_fifo_capability_report.json"
            report.parent.mkdir(parents=True, exist_ok=True)
            report.write_text(json.dumps({
                "cases": [{
                    "case_id": "b001", "native_fifo_supported": True,
                    "native_fifo_ok": False,
                }],
            }), encoding="utf-8")
        elif "native_hailo_trt_fifo_from_benchmarkset.py" in " ".join(command):
            assert len(preparations) == 1
            assert '--no-build' in command and '--no-run' not in command
            result = (
                benchmark_set / "native_pipeline/b001/hailo_to_trt/"
                "uint8_cast_fp16/native_fifo_results.json"
            )
            result.parent.mkdir(parents=True, exist_ok=True)
            result.write_text(json.dumps({
                "ok": True, "fps_makespan": 1.0, "setup_id": "",
            }), encoding="utf-8")
        return {
            "rc": 0, "elapsed_s": 0.01, "stdout_tail": "",
            "stderr_tail": "", "timed_out": False,
        }

    monkeypatch.setattr(matrix, "_run", fake_run)
    monkeypatch.setattr(sys, "argv", [
        "native_fifo_smoke_matrix.py",
        "--benchmark-set", str(benchmark_set),
        "--model-id", "resnet50", "--case", "b001",
        "--image", str(image), "--no-validate-dumps",
    ])
    assert matrix.main() == 0
    result = json.loads((
        benchmark_set / "analysis_tables/native_fifo_smoke_matrix.json"
    ).read_text(encoding="utf-8"))
    assert result["native_split_quality_required"] is False
    assert result["performance_claims_emitted"] is False
    assert result["execution_role"] == "legacy_manual_diagnostic"
    assert result["cases"][0]["result_ok"] is True
    assert result["cases"][0]["performance_claims_emitted"] is False
    assert result['cases'][0]['steps'][-2]['name'] == 'native_wrapper_prepare'


def test_managed_direct_cli_with_quality_set_never_falls_back_to_legacy(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    coordinator = _load_script("native_producer_e2e_eval_runner.py")
    root = tmp_path / "evaluation"
    benchmark_set = root / "resnet50/benchmark_set"
    (benchmark_set / "b001").mkdir(parents=True)
    (benchmark_set / "benchmark_set.json").write_text(json.dumps({
        "task": "classification", "cases": [{"case_id": "b001"}],
    }), encoding="utf-8")
    binding_set = tmp_path / "bindings.json"
    binding_set.write_text(json.dumps({
        "schema": "onnx-splitpoint/native-split-quality-binding-set",
        "schema_version": 1, "setup_id": "deepx_setup",
        "eval_run_id": "eval-qf", "bindings_by_model_case_backend": {},
    }), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", [
        "native_producer_e2e_eval_runner.py", "--root", str(root),
        "--backend", "deepx", "--models", "resnet50",
        "--case-map", '{"resnet50":["b001"]}',
        "--setup-id", "deepx_setup",
        "--native-split-quality-binding-set", str(binding_set),
    ])
    # A supplied but structurally incomplete managed binding set is rejected
    # before any case can fall back to the historical manual path.
    # v31 preserves the selected failed job instead of aborting with rows=0.
    monkeypatch.setattr(coordinator.subprocess, "run", lambda *args, **kwargs: pytest.fail("invalid binding must not launch a child"))
    assert coordinator.main() == 3
    payload = json.loads((root / "analysis_tables/native_deepx_producer_e2e_eval.json").read_text())
    assert len(payload["rows"]) == 1
    row = payload["rows"][0]
    assert row["setup_id"] == "deepx_setup" and row["case"] == "b001"
    assert row["failure_reason"] == "native_split_quality_binding_set_invalid"
    assert row["repetition_count_attempted"] == 0 and row["ok"] is False


def _authority_run(
    root: Path, *, workflow: str, required: bool,
) -> Path:
    reports = root / "reports"
    reports.mkdir(parents=True)
    (root / "run_manifest.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/evaluation-run-manifest",
        "schema_version": 1, "run_id": root.name,
        "workflow_version": workflow,
        "current_workflow_version": workflow,
        "current_tool_version": "2.69.6",
        "execution_sessions": [{"workflow_version": workflow}],
        "profile_start_snapshot": {
            "snapshot_sha256": "b" * 64,
            "requested_selection": {"snapshot_sha256": "a" * 64},
            "resolved_selection": {"snapshot_sha256": "a" * 64},
        },
    }), encoding="utf-8")
    (reports / "native_producer_stage.json").write_text(json.dumps({
        "schema": "onnx-splitpoint/native-producer-stage",
        "schema_version": 1, "run_id": root.name,
        "workflow_version": workflow,
        "profile_start_snapshot_sha256": "b" * 64,
        "profile_selection_snapshot_sha256": "a" * 64,
        "native_split_quality_first": {"required": required},
    }), encoding="utf-8")
    return reports / "native_producer_summary.json"


def test_energy_authority_blocks_marker_stripping_and_limits_legacy_to_diagnostic(
    tmp_path: Path,
) -> None:
    energy = _load_script("native_producer_energy_plan.py")
    current_summary = _authority_run(
        tmp_path / "current-run",
        workflow="v2.69f-hardware-smoke-native-energy-repair",
        required=True,
    )
    current = energy._energy_split_quality_authority(current_summary)
    assert current["valid"] is True and current["mode"] == "required"
    stripped_row = {"backend": "hailo8_to_trt"}
    energy.apply_native_split_quality_authority(stripped_row, current)
    rejected, status = energy._split_quality_energy_evidence(
        stripped_row, {}, current,
    )
    assert rejected is None
    assert status != "legacy_split_quality_first_not_required"
    assert "quality" in status

    legacy_summary = _authority_run(
        tmp_path / "legacy-run", workflow="v2.69e-historical", required=False,
    )
    legacy = energy._energy_split_quality_authority(legacy_summary)
    assert legacy["valid"] is True and legacy["mode"] == "legacy"
    evidence, status = energy._split_quality_energy_evidence(
        {"backend": "hailo8_to_trt"}, {}, legacy,
    )
    assert status == "legacy_split_quality_first_not_required"
    assert evidence is not None
    assert evidence["historical_diagnostic_only"] is True
    assert energy._historical_energy_diagnostic_only(legacy) is True
