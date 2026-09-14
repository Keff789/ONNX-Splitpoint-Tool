from __future__ import annotations

import copy
import importlib.util
import json
import os
from pathlib import Path
import shlex
import sys
from typing import Any

import pytest

from onnx_splitpoint_tool.validation.host_postprocess import (
    resolve_host_postprocess_evidence,
)
from onnx_splitpoint_tool.native_energy_reporting import (
    _normalized_calibration_axes,
    collect_native_energy,
)
from onnx_splitpoint_tool.native_energy_quality_admission import (
    canonical_json_sha256,
)
from scripts import native_producer_final_report as final_report
from tests.test_v2721_completed_v2_host_consumer import (
    _raw_head_completed_v2_row,
)
from tests.test_v270l_completed_endpoint_v2 import (
    _completed_direct_endpoint_evidence,
)
from tests.test_v2721_energy_contract_repairs import (
    _run_full_smoke_plan,
)
from tests.test_v269c_energy_identity_repetition import (
    _energy_payload,
    _plan as _legacy_energy_plan_row,
    _validation as _legacy_energy_validation,
)


ROOT = Path(__file__).resolve().parents[1]
OVERNIGHT_TOKEN = os.environ.get(
    "ONNX_SPLITPOINT_V2722_OVERNIGHT_DEBUG_ROOT", ""
).strip()
OVERNIGHT_ROOT = Path(OVERNIGHT_TOKEN) if OVERNIGHT_TOKEN else None
OVERNIGHT_OBSERVATIONS_RELATIVE = (
    Path("01_results") / "native_performance_observations.json"
)
OVERNIGHT_OBSERVATIONS_CANONICAL = (
    Path("reports")
    / "scientific"
    / "native_performance_observations.json"
)


def _overnight_observations_path(debug_root: Path) -> Path:
    canonical = debug_root / OVERNIGHT_OBSERVATIONS_CANONICAL
    run_bound_candidates: list[Path] = []
    if debug_root.name.endswith("_debug_pack"):
        analysis_pack = debug_root.with_name(
            f"{debug_root.name}_analysis_pack"
        )
        run_bound_candidates.extend(
            (
                analysis_pack / OVERNIGHT_OBSERVATIONS_RELATIVE,
                analysis_pack
                / "analysis"
                / OVERNIGHT_OBSERVATIONS_RELATIVE,
            )
        )
    fallback_candidates = (
        debug_root.parent
        / "analysis"
        / OVERNIGHT_OBSERVATIONS_RELATIVE,
        debug_root / "analysis" / OVERNIGHT_OBSERVATIONS_RELATIVE,
        debug_root / OVERNIGHT_OBSERVATIONS_RELATIVE,
    )

    run_bound_existing = [
        path for path in run_bound_candidates if path.is_file()
    ]
    if canonical.is_file():
        canonical_bytes = canonical.read_bytes()
        divergent = [
            path
            for path in run_bound_existing
            if path.read_bytes() != canonical_bytes
        ]
        if divergent:
            divergent_text = "\n".join(
                f"  - {path}" for path in divergent
            )
            raise AssertionError(
                "run-bound analysis observations diverge from the "
                f"canonical debug report {canonical}:\n"
                f"{divergent_text}"
            )
        return canonical

    if run_bound_existing:
        reference = run_bound_existing[0]
        reference_bytes = reference.read_bytes()
        divergent = [
            path
            for path in run_bound_existing[1:]
            if path.read_bytes() != reference_bytes
        ]
        if divergent:
            divergent_text = "\n".join(
                f"  - {path}" for path in divergent
            )
            raise AssertionError(
                "run-bound analysis observation mirrors diverge:\n"
                f"{divergent_text}"
            )
        return reference

    for candidate in fallback_candidates:
        if candidate.is_file():
            return candidate

    checked = [
        canonical,
        *run_bound_candidates,
        *fallback_candidates,
    ]
    checked_text = "\n".join(f"  - {path}" for path in checked)
    raise FileNotFoundError(
        "could not resolve native performance observations for "
        f"debug pack {debug_root}; checked:\n{checked_text}"
    )


@pytest.mark.parametrize("wrapped", (False, True), ids=("direct", "wrapped"))
def test_overnight_observations_resolver_accepts_analysis_pack_layouts(
    tmp_path: Path,
    wrapped: bool,
) -> None:
    debug_root = tmp_path / "campaign_debug_pack"
    debug_root.mkdir()
    analysis_root = tmp_path / "campaign_debug_pack_analysis_pack"
    if wrapped:
        analysis_root = analysis_root / "analysis"
    observations = analysis_root / OVERNIGHT_OBSERVATIONS_RELATIVE
    observations.parent.mkdir(parents=True)
    observations.write_text("[]\n", encoding="utf-8")

    assert _overnight_observations_path(debug_root) == observations


def test_overnight_observations_resolver_accepts_combined_replay_layout(
    tmp_path: Path,
) -> None:
    debug_root = tmp_path / "debug"
    debug_root.mkdir()
    observations = (
        tmp_path / "analysis" / OVERNIGHT_OBSERVATIONS_RELATIVE
    )
    observations.parent.mkdir(parents=True)
    observations.write_text("[]\n", encoding="utf-8")

    assert _overnight_observations_path(debug_root) == observations


def test_overnight_observations_resolver_prefers_canonical_debug_report(
    tmp_path: Path,
) -> None:
    debug_root = tmp_path / "campaign_debug_pack"
    debug_root.mkdir()
    canonical_observations = (
        debug_root / OVERNIGHT_OBSERVATIONS_CANONICAL
    )
    canonical_observations.parent.mkdir(parents=True)
    canonical_observations.write_text("[]\n", encoding="utf-8")
    analysis_observations = (
        tmp_path
        / "campaign_debug_pack_analysis_pack"
        / OVERNIGHT_OBSERVATIONS_RELATIVE
    )
    analysis_observations.parent.mkdir(parents=True)
    analysis_observations.write_text("[]\n", encoding="utf-8")

    assert (
        _overnight_observations_path(debug_root)
        == canonical_observations
    )


def test_overnight_observations_resolver_rejects_divergent_run_mirror(
    tmp_path: Path,
) -> None:
    debug_root = tmp_path / "campaign_debug_pack"
    debug_root.mkdir()
    canonical_observations = (
        debug_root / OVERNIGHT_OBSERVATIONS_CANONICAL
    )
    canonical_observations.parent.mkdir(parents=True)
    canonical_observations.write_text("[1]\n", encoding="utf-8")
    analysis_observations = (
        tmp_path
        / "campaign_debug_pack_analysis_pack"
        / OVERNIGHT_OBSERVATIONS_RELATIVE
    )
    analysis_observations.parent.mkdir(parents=True)
    analysis_observations.write_text("[2]\n", encoding="utf-8")

    with pytest.raises(AssertionError, match="diverge"):
        _overnight_observations_path(debug_root)


def test_overnight_observations_resolver_prefers_run_sibling(
    tmp_path: Path,
) -> None:
    debug_root = tmp_path / "campaign_debug_pack"
    debug_root.mkdir()
    run_observations = (
        tmp_path
        / "campaign_debug_pack_analysis_pack"
        / OVERNIGHT_OBSERVATIONS_RELATIVE
    )
    run_observations.parent.mkdir(parents=True)
    run_observations.write_text("[]\n", encoding="utf-8")
    unrelated_observations = (
        tmp_path / "analysis" / OVERNIGHT_OBSERVATIONS_RELATIVE
    )
    unrelated_observations.parent.mkdir(parents=True)
    unrelated_observations.write_text("[]\n", encoding="utf-8")

    assert _overnight_observations_path(debug_root) == run_observations


def _load_script(name: str):
    path = ROOT / "scripts" / name
    spec = importlib.util.spec_from_file_location(
        f"v2722_{path.stem}", path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sealed_quality_row(
    *,
    central: bool = True,
    binding: bool = True,
    observation: bool = True,
    provenance: bool = True,
    accuracy: bool = False,
    diagnostic: bool = True,
) -> dict[str, Any]:
    from onnx_splitpoint_tool.native_energy_quality_admission import (
        canonical_json_sha256,
    )

    admission = {
        "schema": "onnx-splitpoint/native-energy-quality-admission",
        "schema_version": 1,
        "backend": "native_full_deepx",
        "model": "yolo26s",
        "case": "full",
        "setup_id": "orin_nx_deepx_m1_01",
        "precision": "fp16",
        "comparison_backend": "deepx",
        "successful_command_contract_sha256": "1" * 64,
        "central_quality_evidence_verified": central,
        "precision_quality_binding_verified": binding,
        "task_quality_observation_valid": observation,
        "accuracy_gate_pass": accuracy,
        "quality_provenance_complete": provenance,
        "quality_claim_result_verified": accuracy,
        "diagnostic_only": diagnostic,
        "screening_comparable": (
            central and binding and observation and provenance
        ),
        "claim_comparable": bool(accuracy and not diagnostic),
        "energy_claim_eligible": bool(accuracy and not diagnostic),
    }
    admission["admission_sha256"] = canonical_json_sha256(admission)
    return {
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
        "energy_quality_admission_sha256": admission["admission_sha256"],
        "claim_ok": False,
        "semantic_claim_ok": False,
        "claim_eligible": False,
        "eligible_for_energy_results_import": False,
        "eligible_for_scientific_claim": False,
    }


@pytest.mark.parametrize(
    "field",
    (
        "central_quality_evidence_verified",
        "precision_quality_binding_verified",
        "task_quality_observation_valid",
        "quality_provenance_complete",
    ),
)
def test_downstream_claim_import_requires_each_sealed_quality_axis(
    field: str,
) -> None:
    from onnx_splitpoint_tool.native_energy_quality_admission import (
        verify_sealed_energy_quality_admission,
    )

    kwargs = {
        "central": True,
        "binding": True,
        "observation": True,
        "provenance": True,
    }
    kwargs[{
        "central_quality_evidence_verified": "central",
        "precision_quality_binding_verified": "binding",
        "task_quality_observation_valid": "observation",
        "quality_provenance_complete": "provenance",
    }[field]] = False
    row = _sealed_quality_row(**kwargs)

    # This seal protects downstream claim import.  It is deliberately not a
    # Native Energy planner-membership predicate.
    with pytest.raises(ValueError, match=field):
        verify_sealed_energy_quality_admission(row, required=True)


def test_bound_negative_accuracy_remains_diagnostic_and_nonclaimable() -> None:
    from onnx_splitpoint_tool.native_energy_quality_admission import (
        verify_sealed_energy_quality_admission,
    )

    row = _sealed_quality_row(accuracy=False, diagnostic=True)
    digest, status = verify_sealed_energy_quality_admission(
        row, required=True,
    )

    assert digest == row["energy_quality_admission_sha256"]
    assert status == "sealed_energy_quality_admission_verified"
    assert row["accuracy_gate_pass"] is False
    assert row["diagnostic_only"] is True
    assert row["energy_claim_eligible"] is False


def test_sealed_quality_admission_rejects_empty_command_contract_sha() -> None:
    from onnx_splitpoint_tool.native_energy_quality_admission import (
        canonical_json_sha256,
        verify_sealed_energy_quality_admission,
    )

    row = _sealed_quality_row()
    admission = row["energy_quality_admission"]
    admission["successful_command_contract_sha256"] = ""
    admission.pop("admission_sha256")
    admission["admission_sha256"] = canonical_json_sha256(admission)
    row["successful_command_contract_sha256"] = ""
    row["energy_quality_admission_sha256"] = admission[
        "admission_sha256"
    ]

    with pytest.raises(
        ValueError,
        match="energy_quality_admission_command_contract_drift",
    ):
        verify_sealed_energy_quality_admission(row, required=True)


def test_declared_tampered_energy_admission_never_uses_legacy_fallback(
    tmp_path: Path,
) -> None:
    plan = _legacy_energy_plan_row("deepx_to_trt", energy=0.10)
    plan.pop("_energy")
    plan["successful_command_contract_sha256"] = "1" * 64
    admission = {
        "schema": "onnx-splitpoint/native-energy-quality-admission",
        "schema_version": 1,
        "admission_scope": "native_energy",
        "backend": plan["backend"],
        "model": plan["model"],
        "case": plan["case"],
        "setup_id": plan["setup_id"],
        "precision": plan["precision"],
        "comparison_backend": plan["comparison_backend"],
        "successful_command_contract_sha256": "1" * 64,
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_provenance_complete": True,
        "quality_claim_result_verified": True,
        "diagnostic_only": False,
        "screening_comparable": True,
        "claim_comparable": True,
        "energy_claim_eligible": True,
    }
    admission["admission_sha256"] = canonical_json_sha256(admission)
    plan.update({
        "energy_quality_admission": admission,
        "energy_quality_admission_sha256": admission[
            "admission_sha256"
        ],
        "central_quality_evidence_verified": True,
        "precision_quality_binding_verified": True,
        "task_quality_observation_valid": True,
        "accuracy_gate_pass": True,
        "quality_claim_result_verified": True,
    })
    # Keep every duplicated legacy axis positive, but invalidate the explicit
    # seal.  The consumer must not fall back to those aliases.
    admission["accuracy_gate_pass"] = False

    validation = _legacy_energy_validation(plan)
    validation_path = (
        tmp_path / "reports" / "native_validation"
        / "native_producer_validation_summary.json"
    )
    validation_path.parent.mkdir(parents=True)
    validation_path.write_text(
        json.dumps({"rows": [validation]}), encoding="utf-8",
    )
    result_path = (
        tmp_path / "reports" / "native_energy_measurements"
        / "native_producer_energy_results.json"
    )
    result_path.parent.mkdir(parents=True)
    result_path.write_text(json.dumps({
        "rows": [{
            "row": plan,
            "ok": True,
            "run": {
                "rc": 0,
                "stdout_tail": _energy_payload(0.10),
            },
        }],
    }), encoding="utf-8")

    [row] = collect_native_energy(tmp_path)

    assert row["energy_quality_admission_verified"] is False
    assert row["energy_quality_admission_status"] == (
        "energy_quality_admission_sha256_mismatch"
    )
    assert row["precision_quality_binding_verified"] is False
    assert row["task_quality_observation_valid"] is False
    assert row["accuracy_gate_pass"] is False
    assert row["quality_provenance_complete"] is False
    assert row["quality_verified"] is False
    assert row["claim_eligible"] is False
    assert (
        "energy_quality_admission_sha256_mismatch"
        in row["claim_exclusion_reasons"]
    )


def test_completed_v2_projection_preserves_the_verified_execution_closure(
) -> None:
    row, _ = _raw_head_completed_v2_row()

    projected = final_report._claim_contract_fields(row)
    resolved = resolve_host_postprocess_evidence(projected)

    assert projected["completion_execution_contract"] == (
        row["completion_execution_contract"]
    )
    assert projected["completion_execution_attestation"] == (
        row["completion_execution_attestation"]
    )
    assert projected["completed_work_units"] == 1
    assert projected["completed_frames"] == 1
    assert projected["completion_observation_relation"] == (
        "same_hotloop_sentinel"
    )
    assert projected["completion_exact_result_claim_bound"] is True
    assert resolved["available"] is True
    assert resolved["status"] == "passed"


def test_validator_preserves_and_strictly_verifies_completed_v2_projection(
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    row, _ = _raw_head_completed_v2_row()
    projected: dict[str, Any] = {}

    validator._copy_completed_v2_projection(projected, row)

    assert projected["completion_execution_contract"] == (
        row["completion_execution_contract"]
    )
    assert projected["completion_execution_attestation"] == (
        row["completion_execution_attestation"]
    )
    assert projected["completion_artifact_sha256"] == (
        row["completion_artifact_sha256"]
    )
    assert validator._completed_execution_attestation_passed(
        projected
    ) is True

    projected["completion_content_sha256"] = "0" * 64
    assert validator._completed_execution_attestation_passed(
        projected
    ) is False


def test_validator_does_not_normalize_away_malformed_completed_v2_alias(
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    row, _ = _raw_head_completed_v2_row()
    row["completion_execution_contract"] = "malformed"
    projected: dict[str, Any] = {}

    validator._copy_completed_v2_projection(projected, row)

    assert projected["completion_execution_contract"] == "malformed"
    assert validator._completed_execution_attestation_passed(
        projected
    ) is False


def test_validator_preserves_direct_bn6_normalization_conflicts(
    tmp_path: Path,
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    row, _ = _completed_direct_endpoint_evidence(tmp_path)
    workload = row["full_command_contract"]["energy_workload"]
    direct_fields = (
        "normalization_frozen",
        "frozen_decoded_nms_normalization_contract",
        "frozen_decoded_nms_normalization_contract_sha256",
        "frozen_decoded_nms_normalization_result",
        "direct_bn6_completion_projection_status",
    )
    row.update({
        "normalization_frozen": True,
        "frozen_decoded_nms_normalization_contract": copy.deepcopy(
            workload["frozen_decoded_nms_normalization_contract"]
        ),
        "frozen_decoded_nms_normalization_contract_sha256": workload[
            "frozen_decoded_nms_normalization_contract_sha256"
        ],
        "frozen_decoded_nms_normalization_result": copy.deepcopy(
            row["completed_task_endpoint_attestation"][
                "frozen_decoded_nms_normalization_result"
            ]
        ),
        "direct_bn6_completion_projection_status": (
            "strict_direct_bn6_completion_verified"
        ),
    })

    def projected(source: dict[str, Any]) -> dict[str, Any]:
        target = copy.deepcopy(source)
        for field in direct_fields:
            target.pop(field, None)
        validator._copy_completed_v2_projection(target, source)
        return target

    valid = projected(row)
    for field in direct_fields:
        assert valid[field] == row[field]
    assert validator._verified_completed_v2_contract(valid)[0] == (
        "integrated_accelerator_plus_frozen_normalization"
    )

    tampered = copy.deepcopy(row)
    tampered["frozen_decoded_nms_normalization_contract"] = {
        "tampered": True,
    }
    with pytest.raises(
        Exception,
        match="completed_v2_direct_normalization_contract_conflict",
    ):
        validator._verified_completed_v2_contract(projected(tampered))


def test_validator_preserves_explicit_host_postprocess_requirement(
) -> None:
    validator = _load_script("native_producer_validate_visualize.py")
    row = {
        "task": "detection",
        "stage": "decoded_nms",
        "contract_family": "decoded_nms",
        "host_postprocess_frozen": False,
        "host_postprocess_required": True,
        "host_tail_required": True,
    }
    projected = copy.deepcopy(row)
    projected.pop("host_postprocess_required")
    projected.pop("host_tail_required")

    validator._copy_completed_v2_projection(projected, row)
    resolved = resolve_host_postprocess_evidence(projected)

    assert projected["host_postprocess_required"] is True
    assert projected["host_tail_required"] is True
    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_incomplete_canonical_host_postprocess_evidence"
    )


def test_known_hailo8_legacy_completion_mode_is_migrated_only_after_strict_verify(
) -> None:
    row, _ = _raw_head_completed_v2_row()
    row["completed_task_completion_mode"] = (
        "completed_detection_execution_contract"
    )
    row.pop("completed_task_endpoint_attestation")
    row.pop("completed_task_endpoint_attested")
    row.pop("completed_task_endpoint_attestation_status")

    projected = final_report._claim_contract_fields(row)
    resolved = resolve_host_postprocess_evidence(projected)

    assert projected["completed_task_completion_mode"] == (
        "detection_completion_execution_v1"
    )
    assert projected["completed_task_endpoint_attestation"] == (
        row["completion_execution_attestation"]
    )
    assert projected["completed_task_endpoint_attested"] is True
    assert projected["completed_task_endpoint_attestation_status"] == "passed"
    assert resolved["available"] is True
    assert resolved["status"] == "passed"


def test_known_hailo8_legacy_completion_tamper_still_fails_closed() -> None:
    row, _ = _raw_head_completed_v2_row()
    row["completed_task_completion_mode"] = (
        "completed_detection_execution_contract"
    )
    row.pop("completed_task_endpoint_attestation")
    row.pop("completed_task_endpoint_attested")
    row.pop("completed_task_endpoint_attestation_status")
    row["completion_execution_attestation"] = copy.deepcopy(
        row["completion_execution_attestation"]
    )
    row["completion_execution_attestation"]["content_sha256"] = "0" * 64

    projected = final_report._claim_contract_fields(row)
    resolved = resolve_host_postprocess_evidence(projected)

    assert projected["completed_task_completion_mode"] == (
        "detection_completion_execution_v1"
    )
    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_invalid_completed_detection_execution_v1"
    )


@pytest.mark.parametrize(
    "field",
    (
        "completion_artifact_sha256",
        "completion_schema_sha256",
        "completion_content_sha256",
        "completion_invocation_sha256",
        "completion_relation_sha256",
    ),
)
def test_completed_v2_rejects_every_tampered_top_level_hash(
    field: str,
) -> None:
    row, _ = _raw_head_completed_v2_row()
    row[field] = "0" * 64

    resolved = resolve_host_postprocess_evidence(row)

    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_invalid_completed_detection_execution_v1"
    )


@pytest.mark.parametrize(
    "field",
    (
        "completion_execution_attestation",
        "completion_execution_contract",
    ),
)
def test_completed_v2_rejects_malformed_alias_next_to_valid_source(
    field: str,
) -> None:
    row, _ = _raw_head_completed_v2_row()
    valid = copy.deepcopy(row[field])
    row[field] = "malformed"

    resolved = resolve_host_postprocess_evidence(
        row,
        {field: valid},
    )

    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_invalid_completed_detection_execution_v1"
    )


def test_completed_v2_projection_rejects_conflicting_sources() -> None:
    row, _ = _raw_head_completed_v2_row()
    conflicting = {
        "completion_content_sha256": "0" * 64,
    }

    projected = final_report._claim_contract_fields(
        row, conflicting,
    )
    resolved = resolve_host_postprocess_evidence(projected)

    assert projected["completed_v2_projection_conflicts"] == [
        "completion_content_sha256"
    ]
    assert resolved["available"] is False
    assert resolved["status"] == (
        "failed_invalid_completed_detection_execution_v1"
    )


def test_hailo8_completed_runtime_scope_is_explicitly_claimable() -> None:
    assert (
        "fresh_hailo_vstreams_trt_completion_runtime_per_repetition"
        in final_report._CLAIMABLE_REPETITION_RUNTIME_SCOPES
    )
    source = (
        ROOT / "onnx_splitpoint_tool" / "native_performance_reporting.py"
    ).read_text(encoding="utf-8")
    assert (
        '"fresh_hailo_vstreams_trt_completion_runtime_per_repetition"'
        in source
    )
    assert "startswith(\"fresh_\")" not in source


def test_sparse_classification_host_postprocess_is_not_required() -> None:
    resolved = resolve_host_postprocess_evidence({
        "task": "classification",
        "stage": "classification_logits",
        "contract_family": "classification_logits",
        "postprocess_included": False,
    })

    assert resolved["available"] is None
    assert resolved["status"] == "not_required"
    assert resolved["source"] == (
        "explicit_host_postprocess_not_required"
    )


def test_classification_detection_task_conflict_fails_closed() -> None:
    resolved = resolve_host_postprocess_evidence(
        {
            "task": "classification",
            "stage": "classification_logits",
            "contract_family": "classification_logits",
            "postprocess_included": False,
        },
        {
            "task": "detection",
            "accelerator_output_stage": "decoded_nms",
            "accelerator_output_contract_family": "decoded_nms",
            "host_postprocess_required": False,
        },
    )

    assert resolved["available"] is False
    assert resolved["status"] != "not_required"


def test_mb_and_full_system_calibration_axes_remain_orthogonal() -> None:
    assert _normalized_calibration_axes(
        "MB",
        verified=False,
        status="missing",
    ) == (
        False,
        "not_applicable_mb",
        "not_required_non_full_system",
    )
    assert _normalized_calibration_axes(
        "full_system",
        verified=False,
        status="missing",
    ) == (
        True,
        "missing_required_full_system_calibration",
        "missing_required_full_system_calibration",
    )
    assert _normalized_calibration_axes(
        "full_system",
        verified=False,
        status="not_required",
    ) == (
        True,
        "missing_required_full_system_calibration",
        "missing_required_full_system_calibration",
    )
    assert _normalized_calibration_axes(
        "unknown",
        verified=False,
        status="missing",
    ) == (
        True,
        "scope_unknown_calibration_unresolved",
        "scope_unknown_calibration_unresolved",
    )


def test_runner_refreshes_expected_matrix_before_any_energy_execution() -> None:
    source = (
        ROOT / "onnx_splitpoint_tool" / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    refresh = source.index(
        "# Refresh the Native denominator for every downstream Energy consumer."
    )
    energy = source.index('label="energy:measure"')
    final_refresh = source.index(
        "expected_matrix = _native_expected_matrix_status_v60y(",
        refresh,
    )

    assert refresh < energy
    assert final_refresh < energy


def test_blocked_nonempty_energy_plan_starts_no_measurement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    out = tmp_path / "energy"
    plan_payload = {
        "rows": [
            {
                "backend": "hailo8_to_trt",
                "model": "resnet50",
                "case": "b001",
                "setup_id": "h8",
                "comparison_backend": "hailo8",
                "precision": "fp16",
            },
            {
                "backend": "hailo8_to_trt",
                "model": "resnet50",
                "case": "b002",
                "setup_id": "h8",
                "comparison_backend": "hailo8",
                "precision": "fp16",
            },
        ],
        "preflight_status": "blocked_energy_coverage_contract_invalid",
        "preflight": {
            "status": "blocked_energy_coverage_contract_invalid",
            "ok": False,
            "measurement_start_allowed": False,
            "energy_plan_coverage_contract_valid": False,
        },
    }
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        calls.append(str(cmd))
        plan_dir = Path(cmd[cmd.index("--out-dir") + 1])
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "native_producer_energy_plan.json").write_text(
            json.dumps(plan_payload), encoding="utf-8",
        )
        return {"rc": 0}

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner.__file__),
            "--summary",
            str(summary),
            "--out-dir",
            str(out),
        ],
    )

    assert runner.main() != 0
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(calls) == 1
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0
    assert len(report["result_ledger"]) == 2
    assert all(
        row["status"] == "blocked_before_measurement"
        for row in report["result_ledger"]
    )


def test_one_technically_invalid_plan_row_blocks_the_complete_cohort(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    summary = tmp_path / "summary.json"
    summary.write_text(json.dumps({"rows": []}), encoding="utf-8")
    out = tmp_path / "energy"
    rows = [
        {
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b001",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "precision": "fp16",
        },
        {
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": "b002",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "precision": "fp16",
        },
    ]
    plan_payload = {
        "rows": rows,
        "preflight_status": "passed",
        "technical_measurement_contract_valid": True,
        "preflight": {
            "status": "passed",
            "ok": True,
            "measurement_start_allowed": True,
            "technical_measurement_contract_valid": True,
            "energy_plan_coverage_contract_valid": True,
        },
    }
    calls: list[str] = []

    def fake_run(cmd, **_kwargs):
        calls.append(str(cmd))
        plan_dir = Path(cmd[cmd.index("--out-dir") + 1])
        plan_dir.mkdir(parents=True, exist_ok=True)
        (plan_dir / "native_producer_energy_plan.json").write_text(
            json.dumps(plan_payload), encoding="utf-8",
        )
        return {"rc": 0}

    def fake_prepare(row, _payload, **kwargs):
        assert kwargs["validate_only"] is True
        if row["case"] == "b002":
            raise ValueError(
                "measure_command_technical_artifact_binding_invalid"
            )
        return {"validated": True}

    monkeypatch.setattr(runner, "_run", fake_run)
    monkeypatch.setattr(
        runner, "_prepare_measurement_execution", fake_prepare,
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(runner.__file__),
            "--summary",
            str(summary),
            "--out-dir",
            str(out),
        ],
    )

    assert runner.main() != 0
    report = json.loads(
        (out / "native_producer_energy_results.json").read_text(
            encoding="utf-8"
        )
    )
    assert len(calls) == 1
    assert report["status"] == "blocked_energy_plan_cohort_invalid"
    assert len(report["rows"]) == 2
    assert len(report["result_ledger"]) == 2
    assert report["measurement_wrapper_started_count"] == 0
    assert report["started_measurement_count"] == 0
    assert report["collector_started_repeat_count"] == 0
    assert report["workload_started_repeat_count"] == 0


def test_incomplete_energy_identity_is_rejected_before_output_state(
    tmp_path: Path,
) -> None:
    runner = _load_script("run_native_producer_energy_from_summary.py")
    allowed = tmp_path / "native_energy"
    row = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b052",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        # precision is deliberately absent
    }

    with pytest.raises(
        ValueError,
        match="energy result identity is incomplete",
    ):
        runner._prepare_measurement_execution(
            row,
            {},
            allowed_root=allowed,
            validate_only=True,
        )

    assert not allowed.exists()


def test_validate_only_rejects_output_outside_evaluation_run(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plan = _run_full_smoke_plan(
        tmp_path,
        monkeypatch,
        binding_verified=True,
        accuracy_gate_pass=False,
    )
    [row] = plan["rows"]
    runner = _load_script("run_native_producer_energy_from_summary.py")
    escaped = tmp_path.parent / f"{tmp_path.name}_escaped"
    parts = shlex.split(row["measure_command"])
    parts = runner._replace_command_option(
        parts, "--out", str(escaped),
    )
    row["measure_command"] = shlex.join(parts)
    row["measurement_output_base_dir"] = str(escaped)
    row["measurement_output_dir"] = str(escaped)

    with pytest.raises(
        ValueError,
        match="escapes EvaluationRun",
    ):
        runner._prepare_measurement_execution(
            row,
            plan,
            allowed_root=tmp_path / "energy-plan",
            validate_only=True,
        )
    assert not escaped.exists()


def test_duplicate_energy_identities_invalidate_every_coverage_claim() -> None:
    from onnx_splitpoint_tool.workflow.evidence_status import (
        derive_native_evidence_status,
    )

    identity = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b001",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "precision": "fp16",
    }
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix={
            "expected_row_count": 2,
            "present_expected_row_count": 2,
            "successful_expected_row_count": 2,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_chain_complete": True,
            "rows": [
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                },
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                },
            ],
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [dict(identity), dict(identity)],
            "excluded_rows": [],
        },
        energy_results_payload={
            "rows": [
                {"row": dict(identity), "ok": True},
                {"row": dict(identity), "ok": True},
            ],
        },
    )

    assert evidence["energy_plan_ledger_valid"] is False
    assert evidence["energy_result_ledger_valid"] is False
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["final_all_split_energy_complete"] is False
    assert evidence["scientific_ready"] is False


def test_expected_energy_identity_cannot_be_replaced_with_same_count_row() -> None:
    from onnx_splitpoint_tool.workflow.evidence_status import (
        derive_native_evidence_status,
    )

    expected_identity = {
        "backend": "hailo8_to_trt",
        "model": "expected-model-a",
        "case": "b001",
        "setup_id": "h8",
        "comparison_backend": "hailo8",
        "precision": "fp16",
    }
    substituted_identity = {
        **expected_identity,
        "model": "unexpected-model-b",
    }
    evidence = derive_native_evidence_status(
        run_mode="final",
        expected_matrix={
            "expected_row_count": 1,
            "present_expected_row_count": 1,
            "successful_expected_row_count": 1,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
            "present_expected_rows": [expected_identity],
            "successful_expected_rows": [expected_identity],
            "failed_expected_rows": [],
            "missing_expected_rows": [],
        },
        validation_payload={"rows": []},
        validation_requested=False,
        energy_requested=True,
        energy_plan_payload={
            "energy_plan_included_count": 1,
            "energy_plan_excluded_count": 0,
            "rows": [substituted_identity],
            "excluded_rows": [],
            "preflight_status": "passed",
            "preflight": {
                "status": "passed",
                "ok": True,
                "measurement_start_allowed": True,
                "energy_plan_coverage_contract_valid": True,
            },
        },
        energy_results_payload={
            "rows": [{
                "row": substituted_identity,
                "ok": True,
            }],
        },
        final_all_split_energy_required=True,
    )

    assert evidence[
        "energy_expected_identity_contract_valid"
    ] is True
    assert evidence["energy_expected_plan_identity_match"] is False
    assert evidence["energy_expected_result_identity_match"] is False
    assert evidence["energy_plan_ledger_valid"] is False
    assert evidence["energy_result_ledger_valid"] is False
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["final_all_split_energy_complete"] is False
    assert evidence["scientific_ready"] is False


def test_full_energy_identity_ignores_legacy_precision_label() -> None:
    from onnx_splitpoint_tool.workflow.evidence_status import (
        derive_native_evidence_status,
    )

    plan_identity = {
        "backend": "native_full_hailo8",
        "model": "resnet50",
        "case": "full",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "precision": "",
    }
    result_identity = {
        **plan_identity,
        "precision": "uint8_cast_fp16",
    }
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix={
            "expected_row_count": 1,
            "present_expected_row_count": 1,
            "successful_expected_row_count": 1,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_chain_complete": True,
            "rows": [{
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": True,
            }],
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [plan_identity],
            "excluded_rows": [],
        },
        energy_results_payload={
            "rows": [{
                "row": result_identity,
                "ok": True,
            }],
        },
    )

    assert evidence["energy_plan_ledger_valid"] is True
    assert evidence["energy_result_ledger_valid"] is True
    assert evidence["energy_coverage_contract_valid"] is True

    duplicate = copy.deepcopy(plan_identity)
    duplicate["precision"] = "fp16"
    duplicate_evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix={
            "expected_row_count": 2,
            "present_expected_row_count": 2,
            "successful_expected_row_count": 2,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_chain_complete": True,
            "rows": [
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                },
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                },
            ],
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [plan_identity, duplicate],
            "excluded_rows": [],
        },
        energy_results_payload={
            "rows": [
                {"row": plan_identity, "ok": True},
                {"row": duplicate, "ok": True},
            ],
        },
    )
    assert duplicate_evidence["energy_plan_ledger_valid"] is False
    assert duplicate_evidence["energy_result_ledger_valid"] is False


def test_result_counts_must_match_result_row_statuses() -> None:
    from onnx_splitpoint_tool.workflow.evidence_status import (
        derive_native_evidence_status,
    )

    identity = {
        "backend": "hailo8_to_trt",
        "model": "resnet50",
        "case": "b001",
        "setup_id": "orin_nx_hailo8_01",
        "comparison_backend": "hailo8",
        "precision": "fp16",
    }
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix={
            "expected_row_count": 1,
            "present_expected_row_count": 1,
            "successful_expected_row_count": 1,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_chain_complete": True,
            "rows": [{
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": True,
            }],
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [identity],
            "excluded_rows": [],
        },
        energy_results_payload={
            "measurement_success_count": 1,
            "measurement_failed_count": 0,
            "rows": [{
                "row": identity,
                "ok": False,
            }],
        },
    )

    assert evidence["energy"]["result_status_counts_consistent"] is False
    assert evidence["energy_result_ledger_valid"] is False
    assert evidence["energy_coverage_contract_valid"] is False
    assert evidence["evidence_complete"] is False


def test_debug_pack_keeps_large_offline_replay_core(
    tmp_path: Path,
) -> None:
    packer = _load_script("create_evaluation_debug_pack.py")
    for relative in packer.REPLAY_CORE:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("x" * 2048, encoding="utf-8")
        included, reason = packer._should_include(
            tmp_path,
            path,
            16,
            probe_include_raw=False,
        )
        assert included is True, (relative, reason)


def test_preflight_block_is_terminal_but_not_a_measurement_failure() -> None:
    from onnx_splitpoint_tool.workflow.evidence_status import (
        derive_native_evidence_status,
    )

    def identity(index: int) -> dict[str, str]:
        return {
            "backend": "hailo8_to_trt",
            "model": "resnet50",
            "case": f"b{index:03d}",
            "setup_id": "h8",
            "comparison_backend": "hailo8",
            "precision": "fp16",
        }

    rows = [identity(1), identity(2)]
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix={
            "expected_row_count": 2,
            "present_expected_row_count": 2,
            "successful_expected_row_count": 2,
            "failed_expected_row_count": 0,
            "missing_expected_row_count": 0,
        },
        validation_payload={
            "technical_chain_complete": True,
            "rows": [
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                },
                {
                    "semantic_available": True,
                    "semantic_ok": True,
                    "claim_ok": True,
                },
            ],
        },
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": rows,
            "excluded_rows": [],
            "preflight_status": "blocked_energy_coverage_contract_invalid",
            "preflight": {
                "status": "blocked_energy_coverage_contract_invalid",
                "ok": False,
                "measurement_start_allowed": False,
                "energy_plan_coverage_contract_valid": False,
            },
        },
        energy_results_payload={
            "rows": [
                {
                    "row": row,
                    "ok": False,
                    "skipped": "blocked_before_measurement",
                    "measurement_started": False,
                }
                for row in rows
            ],
        },
    )

    assert evidence["energy_terminal_result_count"] == 2
    assert evidence["energy_not_started_preflight_count"] == 2
    assert evidence["energy_measurement_started_count"] == 0
    assert evidence["energy_measurement_success_count"] == 0
    assert evidence["energy_measurement_failed_count"] == 0
    assert evidence["energy_terminal_result_ledger_valid"] is True
    assert evidence["energy_coverage_contract_valid"] is False


def test_current_status_projection_does_not_publish_deprecated_green_aliases(
) -> None:
    from onnx_splitpoint_tool.workflow.evidence_status import (
        derive_native_evidence_status,
    )

    expected = {
        "expected_row_count": 2,
        "present_expected_row_count": 2,
        "successful_expected_row_count": 2,
        "failed_expected_row_count": 0,
        "missing_expected_row_count": 0,
    }
    validation = {
        "rows": [
            {
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": True,
            },
            {
                "semantic_available": True,
                "semantic_ok": True,
                "claim_ok": False,
            },
        ],
        "technical_chain_complete": True,
    }
    evidence = derive_native_evidence_status(
        run_mode="standard",
        expected_matrix=expected,
        validation_payload=validation,
        validation_requested=True,
        energy_requested=True,
        energy_plan_payload={
            "rows": [
                {
                    "backend": "hailo8_to_trt",
                    "model": "resnet50",
                    "case": case,
                    "setup_id": "h8",
                    "comparison_backend": "hailo8",
                    "precision": "fp16",
                }
                for case in ("b001", "b002")
            ],
            "excluded_rows": [],
            "energy_plan_included_count": 2,
            "energy_plan_excluded_count": 0,
        },
        energy_results_payload={
            "rows": [
                {
                    "row": {
                        "backend": "hailo8_to_trt",
                        "model": "resnet50",
                        "case": case,
                        "setup_id": "h8",
                        "comparison_backend": "hailo8",
                        "precision": "fp16",
                    },
                    "ok": True,
                    "energy_claim_eligible": False,
                }
                for case in ("b001", "b002")
            ]
        },
    )

    assert evidence["claim_decisions_complete"] is True
    assert evidence["energy_plan_included_count"] == 2
    assert evidence["energy_plan_excluded_count"] == 0
    assert evidence["task_quality"]["pass_count"] == 1
    assert evidence["task_quality"]["fail_count"] == 1
    assert evidence["scientific_ready"] is True
    assert "claim_ready" not in evidence
    assert "energy_complete" not in evidence
    assert "ready" not in evidence["claim"]
    assert "complete" not in evidence["energy"]
    assert evidence["deprecated_aliases"]["claim_ready"]["value"] is True
    assert evidence["deprecated_aliases"]["energy_complete"]["value"] is True


@pytest.mark.skipif(
    OVERNIGHT_ROOT is None or not OVERNIGHT_ROOT.is_dir(),
    reason="set ONNX_SPLITPOINT_V2722_OVERNIGHT_DEBUG_ROOT for replay",
)
def test_real_overnight_completed_v2_rows_survive_projection() -> None:
    assert OVERNIGHT_ROOT is not None
    producers = OVERNIGHT_ROOT / "native_producers"
    cases = (
        (
            final_report._rows_from_deepx,
            producers / "deepx",
            "deepx_to_trt",
            "yolo26s",
            "b036",
        ),
        (
            final_report._rows_from_hailo10,
            producers / "hailo10h",
            "hailo10h_to_trt",
            "yolov7_paper",
            "b044",
        ),
        (
            final_report._rows_from_native_fifo_runner,
            producers / "hailo8",
            "hailo8_to_trt",
            "yolo26s",
            "b036",
        ),
        (
            final_report._rows_from_native_fifo_runner,
            producers / "hailo8",
            "hailo8_to_trt",
            "yolov7_paper",
            "b044",
        ),
    )
    for loader, root, backend, model, case in cases:
        matches = [
            row for row in loader(root)
            if row.get("backend") == backend
            and row.get("model") == model
            and row.get("case") == case
        ]
        assert len(matches) == 1
        [row] = matches
        resolved = resolve_host_postprocess_evidence(row)
        assert row["completed_work_units"] == 1000
        assert row["completed_frames"] == 1000
        assert row["completion_observation_relation"] == (
            "same_hotloop_sentinel"
        )
        assert row["completion_exact_result_claim_bound"] is True
        assert resolved["available"] is True
        assert resolved["status"] == "passed"


@pytest.mark.skipif(
    OVERNIGHT_ROOT is None or not OVERNIGHT_ROOT.is_dir(),
    reason="set ONNX_SPLITPOINT_V2722_OVERNIGHT_DEBUG_ROOT for replay",
)
def test_real_overnight_hailo8_repetition_and_accuracy_values_are_unchanged(
) -> None:
    assert OVERNIGHT_ROOT is not None
    rows = final_report._rows_from_native_fifo_runner(
        OVERNIGHT_ROOT / "native_producers" / "hailo8"
    )
    hailo8 = next(
        row for row in rows
        if row.get("backend") == "hailo8_to_trt"
        and row.get("model") == "yolov7_paper"
        and row.get("case") == "b044"
    )
    valid, reasons = final_report._validate_repeat_claim_evidence(
        copy.deepcopy(hailo8)
    )
    assert valid is True
    assert reasons == []

    unknown_scope = copy.deepcopy(hailo8)
    unknown_scope["repetition_runtime_scope"] = (
        "fresh_unrecognized_runtime_per_repetition"
    )
    valid, reasons = final_report._validate_repeat_claim_evidence(
        unknown_scope
    )
    assert valid is False
    assert "repetition_runtime_scope_not_independent" in reasons

    observations = json.loads(
        _overnight_observations_path(OVERNIGHT_ROOT).read_text(
            encoding="utf-8"
        )
    )
    yolo26 = next(
        row for row in observations
        if row.get("model_id") == "yolo26s"
        and row.get("backend") == "native_full_hailo10h"
    )
    yolov7 = next(
        row for row in observations
        if row.get("model_id") == "yolov7_paper"
        and row.get("backend") == "native_full_hailo10h"
    )
    assert yolo26["numerical_similarity_value"] == (
        0.6666666666666666
    )
    assert yolo26["numerical_similarity_threshold"] == 0.8
    assert yolo26["numerical_similarity_pass"] is False
    assert yolo26["numerical_similarity_reason"] == (
        "reference_match_ratio_below_threshold"
    )
    assert yolov7["numerical_similarity_mean_iou"] == (
        0.8914144932221643
    )
    assert yolov7["numerical_similarity_mean_iou_threshold"] == 0.9
    assert yolov7["numerical_similarity_pass"] is False
    assert yolov7["numerical_similarity_reason"] == (
        "mean_matched_iou_below_threshold"
    )
