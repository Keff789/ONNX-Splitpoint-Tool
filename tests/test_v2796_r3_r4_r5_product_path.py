from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.logical_measurement import (
    annotate_logical_measurements,
    select_logical_primary_rows,
    summarize_logical_measurements,
)
from onnx_splitpoint_tool.workflow.required_run_scope import (
    RequiredRunScopeError,
    authoritative_run_descriptors,
    build_global_required_run_scope,
    build_model_required_run_scope,
    merge_authoritative_run_descriptors,
    required_measurements_from_scope,
    seal_required_run_scope,
)
from onnx_splitpoint_tool.workflow.runner import (
    _authoritative_benchmark_plan_v2792,
    _bind_results_to_required_scope_v2796,
    duplicate_profile_measurements_v269d,
    expected_profile_measurements_v60r,
    missing_profile_measurements_v60r,
)
from onnx_splitpoint_tool.workflow.results import normalize_benchmark_row


REQUEST_SHA = "a" * 64
DXNN_SHA = "b" * 64
SETUP_ID = "orin_nx_deepx_m1_01"


def _required() -> list[dict]:
    measurement = {
        "model_id": "yolov7_paper",
        "case_id": "b066",
        "run_id": "deepx_m1_to_tensorrt",
        "backend": "deepx_m1_to_tensorrt",
        "variant": "split",
        "expected_setup_id": SETUP_ID,
        "measurement_endpoint": "p2_output",
        "quality_endpoint": "completed_detection",
    }
    scope = build_model_required_run_scope(
        model_id="yolov7_paper",
        measurements=[measurement],
        benchmark_plan={
            "runs": [{
                "id": "deepx_m1_to_tensorrt",
                "expected_setup_id": SETUP_ID,
                "measurement_endpoint": "p2_output",
                "quality_endpoint": "completed_detection",
            }],
        },
        benchmark_set_contract={"cases": [{"case_id": "b066"}]},
        created_at="before-compiler",
    )
    return required_measurements_from_scope(scope)


def _result(**overrides: object) -> dict:
    row = {
        "model_id": "yolov7_paper",
        "case_id": "b066",
        "run_id": "deepx_m1_to_tensorrt",
        "source_run_id": "deepx_m1_to_tensorrt",
        "backend": "deepx_m1_to_tensorrt",
        "variant": "split",
        "task": "detection",
        "setup_id": SETUP_ID,
        "measurement_endpoint": "p2_output",
        "quality_endpoint": "completed_detection",
        "source_request_sha256": REQUEST_SHA,
        "dxnn_sha256": DXNN_SHA,
        "source_path": "direct.json",
        "pipeline_fps_selected": 5.0,
    }
    row.update(overrides)
    return row


def _logical_rows(rows: list[dict]) -> tuple[list[dict], list[dict], list[dict]]:
    bound, binding_errors = _bind_results_to_required_scope_v2796(
        _required(), rows,
    )
    annotated = annotate_logical_measurements(bound)
    primaries, group_errors = select_logical_primary_rows(annotated)
    assert group_errors == []
    return annotated, primaries, binding_errors


def test_exact_verified_setup_less_mirror_counts_once_product_path() -> None:
    direct = _result()
    mirror = _result(
        setup_id="", source_path="mirror.json",
        pipeline_fps_selected=None,
    )
    annotated, primaries, binding_errors = _logical_rows([direct, mirror])
    summary = summarize_logical_measurements(annotated)
    assert summary["raw_representation_count"] == 2
    assert summary["logical_measurement_count"] == 1
    assert summary["logical_primary_count"] == 1
    assert summary["verified_mirror_representation_count"] == 1
    assert len(binding_errors) == 1  # supplemental mirror is not physical scope
    assert missing_profile_measurements_v60r(_required(), primaries) == []
    assert duplicate_profile_measurements_v269d(_required(), primaries) == []


def test_unverified_setup_less_representation_is_not_collapsed() -> None:
    direct = _result(source_request_sha256="", dxnn_sha256="")
    mirror = _result(
        setup_id="", source_request_sha256="", dxnn_sha256="",
        source_path="mirror.json", pipeline_fps_selected=None,
    )
    annotated, primaries, _ = _logical_rows([direct, mirror])
    assert len(primaries) == 2
    assert len({row["logical_measurement_id"] for row in annotated}) == 2
    assert summarize_logical_measurements(annotated)[
        "unverified_representation_count"
    ] == 1


def test_two_direct_rows_remain_duplicate_in_product_completeness() -> None:
    first = _result(source_path="one.json")
    second = _result(source_path="two.json")
    _, primaries, _ = _logical_rows([first, second])
    duplicates = duplicate_profile_measurements_v269d(
        _required(), primaries,
    )
    assert len(primaries) == 2
    assert len(duplicates) == 1
    assert duplicates[0]["row_count"] == 2


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("setup_id", "wrong_setup"),
        ("measurement_endpoint", "completed_detection"),
    ],
)
def test_wrong_physical_result_does_not_satisfy_scope(
    field: str, value: str,
) -> None:
    row = _result()
    row[field] = value
    bound, errors = _bind_results_to_required_scope_v2796(
        _required(), [row],
    )
    annotated = annotate_logical_measurements(bound)
    primaries, group_errors = select_logical_primary_rows(annotated)
    assert group_errors == []
    assert errors[0]["error_class"] == "scope_identity_physical_mismatch"
    assert len(missing_profile_measurements_v60r(_required(), primaries)) == 1


def test_scope_receipt_is_immutable_after_result_binding() -> None:
    required = _required()
    before = deepcopy(required)
    _bind_results_to_required_scope_v2796(required, [_result()])
    assert required == before


def test_deepx_runtime_precision_survives_product_normalization() -> None:
    structured = {
        "schema": "onnx-splitpoint/deepx-runtime-precision-contract",
        "artifact_kind": "dxnn",
        "artifact_sha256": DXNN_SHA,
        "precision_semantics": "opaque_vendor_compiled_artifact_identity",
    }
    raw = {
        "model_id": "yolov7_paper",
        "case_id": "full",
        "variant": "full",
        "primary_variant": "full",
        "backend": "deepx_m1",
        "full_backend": "deepx_m1",
        "run_id": "deepx_m1_full",
        "source_run_id": "deepx_m1_full",
        "setup_id": SETUP_ID,
        "total_latency_ms": 1.0,
        "runtime_ok": True,
        "runtime_artifact_sha256": DXNN_SHA,
        "full_runtime_artifact_sha256": DXNN_SHA,
        "runtime_precision_identity": (
            f"deepx_dxnn_sha256:{DXNN_SHA}"
        ),
        "full_runtime_precision_identity": (
            f"deepx_dxnn_sha256:{DXNN_SHA}"
        ),
        "candidate_execution_contract": {
            "runtime_precision_identity": structured,
        },
        "quality_input_request": {
            "runtime_precision_identity": structured,
        },
        "measurement_endpoint": "completed_detection",
        "quality_endpoint": "completed_detection",
    }
    normalized = normalize_benchmark_row(
        raw, model_id="yolov7_paper", source_path=Path("fixture.json"),
    )
    assert normalized["runtime_precision_identity"] == (
        f"deepx_dxnn_sha256:{DXNN_SHA}"
    )
    assert normalized["runtime_artifact_sha256"] == DXNN_SHA
    frozen = normalized["frozen_identity_evidence"]
    assert frozen["runtime_precision_identity_resolution_status"] == (
        "canonical"
    )
    assert frozen["runtime_precision_identity_canonical_candidates"] == [
        f"deepx_dxnn_sha256:{DXNN_SHA}",
    ]
    assert len(
        frozen["runtime_precision_identity_raw_representations"]
    ) == 2


def test_six_run_scope_reaches_model_receipt_with_physical_fields() -> None:
    run_ids = [
        "hailo8", "hailo10", "deepx_m1_full",
        "hailo8_to_trt", "hailo10_to_tensorrt",
        "deepx_m1_to_tensorrt",
    ]
    global_scope = build_global_required_run_scope(
        profile_id="strict",
        model_entries=[{"id": "yolov7_paper", "task": "detection"}],
        effective_plan={
            "logical_run_profiles": run_ids,
            "effective_generic_run_ids": run_ids,
            "setup_groups": {
                "hailo8_setup": ["hailo8", "hailo8_to_trt"],
                "hailo10h_setup": [
                    "hailo10", "hailo10_to_tensorrt",
                ],
                "deepx_setup": [
                    "deepx_m1_full", "deepx_m1_to_tensorrt",
                ],
            },
        },
        hardware_targets=[
            {"id": "h8", "accelerator": "hailo8"},
            {"id": "h10", "accelerator": "hailo10"},
            {"id": "dx", "accelerator": "deepx_m1"},
        ],
        created_at="before-compiler",
    )
    plan = merge_authoritative_run_descriptors(
        {"runs": []},
        authoritative_run_descriptors(
            global_scope, model_id="yolov7_paper",
        ),
    )
    case_contract = {"cases": [{"case_id": "b066"}]}
    measurements = expected_profile_measurements_v60r(
        model_id="yolov7_paper",
        benchmark_plan=plan,
        benchmark_set_contract=case_contract,
    )
    model_scope = build_model_required_run_scope(
        model_id="yolov7_paper",
        measurements=measurements,
        benchmark_plan=plan,
        benchmark_set_contract=case_contract,
        created_at="before-compiler",
        global_scope_sha256="f" * 64,
    )
    assert model_scope["identity_count"] == 6
    assert all(
        row["expected_setup_id"]
        and row["measurement_endpoint"]
        and row["quality_endpoint"]
        and row["logical_identity_sha256"]
        for row in model_scope["identities"]
    )


def test_runner_projects_strict_descriptors_and_legacy_needs_sidecar(
    tmp_path: Path,
) -> None:
    strict = build_global_required_run_scope(
        profile_id="strict",
        model_entries=[{"id": "yolov7_paper", "task": "detection"}],
        effective_plan={
            "logical_run_profiles": ["hailo8"],
            "effective_generic_run_ids": ["hailo8"],
            "setup_groups": {"hailo8_setup": ["hailo8"]},
        },
        hardware_targets=[{"id": "h8", "accelerator": "hailo8"}],
        created_at="before-compiler",
    )
    seal_required_run_scope(tmp_path / "required_run_scope.json", strict)
    plan, evidence = _authoritative_benchmark_plan_v2792(
        tmp_path, {"runs": []}, model_id="yolov7_paper",
    )
    assert evidence["identity_mode"] == "physical_strict_v3"
    assert plan["runs"][0]["setup_id"] == "h8"
    assert plan["runs"][0]["measurement_endpoint"] == (
        "completed_detection"
    )

    legacy_dir = tmp_path / "legacy"
    legacy_dir.mkdir()
    seal_required_run_scope(legacy_dir / "required_run_scope.json", {
        "schema": "onnx-splitpoint/required-run-scope",
        "schema_version": 2,
        "scope_level": "run",
        "requested_run_ids": ["hailo8"],
    })
    with pytest.raises(
        RequiredRunScopeError,
        match="legacy_projection_not_opted_in",
    ):
        _authoritative_benchmark_plan_v2792(
            legacy_dir, {"runs": []}, model_id="yolov7_paper",
        )
    (legacy_dir / "legacy_reprojection_opt_in.json").write_text(
        '{"legacy_reprojection":true,"explicit_opt_in":true}',
        encoding="utf-8",
    )
    legacy_plan, legacy_evidence = _authoritative_benchmark_plan_v2792(
        legacy_dir, {"runs": []}, model_id="yolov7_paper",
    )
    assert legacy_evidence["identity_mode"] == "legacy_reprojection_v2"
    assert legacy_plan["runs"][0]["id"] == "hailo8"
