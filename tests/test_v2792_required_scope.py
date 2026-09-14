from __future__ import annotations

import json
from pathlib import Path

import pytest

from onnx_splitpoint_tool.workflow.required_run_scope import (
    RequiredRunScopeError,
    authoritative_run_descriptors,
    authoritative_run_ids,
    build_global_required_run_scope,
    build_model_required_run_scope,
    merge_authoritative_run_descriptors,
    merge_authoritative_runs,
    seal_required_run_scope,
)


def test_scope_resume_ignores_timestamp_but_not_semantics(tmp_path: Path) -> None:
    path = tmp_path / "required_run_scope.json"
    first = seal_required_run_scope(path, {
        "schema": "onnx-splitpoint/required-run-scope",
        "schema_version": 2,
        "created_at": "first",
        "scope_level": "run",
        "requested_run_ids": ["hailo8", "hailo10"],
    })
    second = seal_required_run_scope(path, {
        "schema": "onnx-splitpoint/required-run-scope",
        "schema_version": 2,
        "created_at": "second",
        "scope_level": "run",
        "requested_run_ids": ["hailo8", "hailo10"],
    })
    assert second == first
    with pytest.raises(RequiredRunScopeError, match="immutable_mismatch"):
        seal_required_run_scope(path, {
            "schema": "onnx-splitpoint/required-run-scope",
            "schema_version": 2,
            "created_at": "third",
            "scope_level": "run",
            "requested_run_ids": ["hailo10"],
        })


def test_global_scope_prevents_model_local_hailo8_shrink() -> None:
    global_scope = build_global_required_run_scope(
        profile_id="audit",
        model_entries=[{"id": "yolo11l", "role": "transfer"}],
        effective_plan={
            "logical_run_profiles": ["hailo8", "hailo10", "deepx_m1_full"],
            "effective_generic_run_ids": ["hailo8", "hailo10", "deepx_m1_full"],
        },
        hardware_targets=[],
        created_at="now",
        legacy_reprojection=True,
    )
    assert authoritative_run_ids(global_scope) == [
        "hailo8", "hailo10", "deepx_m1_full",
    ]
    local = {"model_id": "yolo11l", "runs": [{"id": "hailo10"}]}
    merged = merge_authoritative_runs(local, authoritative_run_ids(global_scope))
    ids = [row["id"] for row in merged["runs"]]
    assert ids == ["hailo10", "hailo8", "deepx_m1_full"]
    assert merged["authoritative_scope_projection"]["injected_run_ids"] == [
        "hailo8", "deepx_m1_full",
    ]


def test_model_scope_keeps_failed_full_as_terminal_required() -> None:
    plan = merge_authoritative_runs(
        {"runs": [{"id": "hailo10"}]}, ["hailo8", "hailo10"],
    )
    measurements = [
        {"model_id": "yolo11l", "case_id": "full", "run_id": "hailo8", "backend": "hailo8", "variant": "full"},
        {"model_id": "yolo11l", "case_id": "full", "run_id": "hailo10", "backend": "hailo10", "variant": "full"},
    ]
    scope = build_model_required_run_scope(
        model_id="yolo11l",
        measurements=measurements,
        benchmark_plan=plan,
        benchmark_set_contract={"cases": []},
        created_at="now",
        global_scope_sha256="a" * 64,
        legacy_reprojection=True,
    )
    assert scope["identity_count"] == 2
    by_run = {row["run_id"]: row for row in scope["identities"]}
    assert by_run["hailo8"]["terminal_outcome_required"] is True
    assert by_run["hailo8"]["success_required"] is False
    assert by_run["hailo8"]["quality_applicability"] == "applicable"


def test_materialized_plan_difference_does_not_rewrite_scope() -> None:
    from onnx_splitpoint_tool.workflow.required_run_scope import (
        audit_materialized_scope,
        required_measurements_from_scope,
    )

    plan = merge_authoritative_runs(
        {"runs": [{"id": "hailo10"}]}, ["hailo8", "hailo10"],
    )
    scope = build_model_required_run_scope(
        model_id="yolo11l",
        measurements=[
            {"model_id": "yolo11l", "case_id": "full", "run_id": "hailo8", "backend": "hailo8", "variant": "full"},
            {"model_id": "yolo11l", "case_id": "full", "run_id": "hailo10", "backend": "hailo10", "variant": "full"},
        ],
        benchmark_plan=plan,
        benchmark_set_contract={"cases": []},
        created_at="before-compiler",
        legacy_reprojection=True,
    )
    materialized = [
        {"model_id": "yolo11l", "case_id": "full", "run_id": "hailo10", "backend": "hailo10", "variant": "full"},
    ]
    audit = audit_materialized_scope(
        scope=scope, materialized_measurements=materialized,
    )
    assert audit["status"] == "difference_recorded"
    assert audit["missing_from_materialized_count"] == 1
    assert audit["missing_from_materialized"][0]["run_id"] == "hailo8"
    assert audit["scope_rewritten"] is False
    required = required_measurements_from_scope(scope)
    assert {row["run_id"] for row in required} == {"hailo8", "hailo10"}


def _strict_global_scope() -> dict:
    run_ids = [
        "hailo8", "hailo10", "deepx_m1_full",
        "hailo8_to_trt", "hailo10_to_tensorrt",
        "deepx_m1_to_tensorrt",
    ]
    return build_global_required_run_scope(
        profile_id="strict",
        model_entries=[{"id": "yolo11l", "task": "detection"}],
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
            {"id": "orin_nx_hailo8_01", "accelerator": "hailo8"},
            {"id": "orin_nx_hailo10_01", "accelerator": "hailo10"},
            {"id": "orin_nx_deepx_m1_01", "accelerator": "deepx_m1"},
        ],
        created_at="before-compiler",
    )


def test_strict_global_scope_seals_six_physical_run_descriptors() -> None:
    scope = _strict_global_scope()
    descriptors = authoritative_run_descriptors(
        scope, model_id="yolo11l",
    )
    assert scope["schema_version"] == 3
    assert scope["identity_mode"] == "physical_strict_v3"
    assert len(descriptors) == 6
    assert all(row["expected_setup_id"] for row in descriptors)
    assert all(row["measurement_endpoint"] for row in descriptors)
    assert all(row["quality_endpoint"] for row in descriptors)
    split = [row for row in descriptors if row["variant"] == "split"]
    assert {row["measurement_endpoint"] for row in split} == {"p2_output"}
    assert {row["quality_endpoint"] for row in split} == {
        "completed_detection",
    }


def test_strict_scope_requires_explicit_legacy_opt_in_for_blank_fields() -> None:
    with pytest.raises(
        RequiredRunScopeError, match="incomplete_physical_scope_identity",
    ):
        build_model_required_run_scope(
            model_id="yolo11l",
            measurements=[{
                "model_id": "yolo11l", "case_id": "full",
                "run_id": "hailo8", "backend": "hailo8",
                "variant": "full",
            }],
            benchmark_plan={"runs": [{"id": "hailo8"}]},
            benchmark_set_contract={"cases": []},
            created_at="now",
        )
    legacy = build_model_required_run_scope(
        model_id="yolo11l",
        measurements=[{
            "model_id": "yolo11l", "case_id": "full",
            "run_id": "hailo8", "backend": "hailo8",
            "variant": "full",
        }],
        benchmark_plan={"runs": [{"id": "hailo8"}]},
        benchmark_set_contract={"cases": []},
        created_at="now",
        legacy_reprojection=True,
    )
    assert legacy["identity_mode"] == "legacy_reprojection_v2"


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("setup_id", "wrong_setup"),
        ("measurement_endpoint", "wrong_endpoint"),
        ("quality_endpoint", "wrong_quality_endpoint"),
    ],
)
def test_conflicting_physical_descriptor_merge_fails_closed(
    field: str, value: str,
) -> None:
    scope = _strict_global_scope()
    descriptors = authoritative_run_descriptors(
        scope, model_id="yolo11l",
    )
    plan = {"runs": [{"id": "hailo8", field: value}]}
    with pytest.raises(
        RequiredRunScopeError, match="physical_descriptor_conflict",
    ):
        merge_authoritative_run_descriptors(plan, descriptors)
