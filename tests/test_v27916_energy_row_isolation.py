from __future__ import annotations

import pytest

from onnx_splitpoint_tool.workflow.runner import (
    _merge_row_local_quality_contracts,
    _native_backend_start_counts,
    _native_energy_not_started_projection,
    _native_global_infrastructure_failure,
    _native_model_preflight_rows,
    _native_remote_launch_observed,
)


def test_partial_trt_quality_handoff_retains_valid_model_rows() -> None:
    base = {
        "schema": "onnx-splitpoint/tensorrt-quality-producer-set",
        "schema_version": 1,
        "eval_run_id": "eval-1",
        "setup_id": "setup-1",
    }
    merged = _merge_row_local_quality_contracts(
        [
            {**base, "producers_by_model": {"resnet50": {"id": "r"}}},
            {**base, "producers_by_model": {"yolo26s": {"id": "y"}}},
        ],
        rows_field="producers_by_model",
    )
    assert merged["producers_by_model"] == {
        "resnet50": {"id": "r"},
        "yolo26s": {"id": "y"},
    }


def test_partial_split_quality_handoff_retains_valid_selection_rows() -> None:
    base = {
        "schema": "onnx-splitpoint/native-split-quality-binding-set",
        "schema_version": 2,
        "eval_run_id": "eval-1",
        "setup_id": "setup-1",
        "central_quality_summary_sha256": "a" * 64,
    }
    merged = _merge_row_local_quality_contracts(
        [
            {
                **base,
                "bindings_by_model_case_backend": {
                    "resnet50|b500|hailo8_to_trt": {"id": "r"},
                },
                "binding_set_sha256": "stale-a",
            },
            {
                **base,
                "bindings_by_model_case_backend": {
                    "yolo26s|b500|hailo8_to_trt": {"id": "y"},
                },
                "binding_set_sha256": "stale-b",
            },
        ],
        rows_field="bindings_by_model_case_backend",
        sha_field="binding_set_sha256",
    )
    assert set(merged["bindings_by_model_case_backend"]) == {
        "resnet50|b500|hailo8_to_trt",
        "yolo26s|b500|hailo8_to_trt",
    }
    assert merged["binding_set_sha256"] not in {"stale-a", "stale-b"}


def test_quality_handoff_rejects_cross_run_or_setup_scope() -> None:
    with pytest.raises(
        ValueError, match="conflicting contract scope",
    ):
        _merge_row_local_quality_contracts(
            [
                {
                    "schema": "quality-set",
                    "schema_version": 1,
                    "eval_run_id": "eval-1",
                    "setup_id": "setup-1",
                    "producers_by_model": {"resnet50": {"id": "r"}},
                },
                {
                    "schema": "quality-set",
                    "schema_version": 1,
                    "eval_run_id": "eval-2",
                    "setup_id": "setup-2",
                    "producers_by_model": {"yolo26s": {"id": "y"}},
                },
            ],
            rows_field="producers_by_model",
        )


def test_valid_and_invalid_model_rows_keep_valid_row_runnable() -> None:
    rows = _native_model_preflight_rows(
        ["resnet50"],
        {"yolov7_ultralytics": {"valid": False}},
        {"resnet50": {"valid": True}},
        energy_requested=True,
    )
    assert [(row["model"], row["status"]) for row in rows] == [
        ("yolov7_ultralytics", "blocked"),
        ("resnet50", "ready"),
    ]
    blocked, ready = rows
    assert blocked["energy_not_started_categories"] == ["model_preflight"]
    assert blocked["energy_not_started_reason"] == (
        "Energy requested, not started: Native preflight blocked by "
        "benchmark_set_invalid:yolov7_ultralytics"
    )
    assert ready["energy_not_started_reason"] == ""


def test_valid_backend_start_counters_are_not_lost() -> None:
    counts = _native_backend_start_counts([{
        "backend": "hailo8",
        "started_remote_count": 2,
        "started_performance_count": 2,
    }])
    assert counts == {
        "started_remote_count": 2,
        "started_performance_count": 2,
    }
    projected = _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "blocked_zero_measurements_started",
            "started_measurement_count": 0,
            **counts,
        },
    )
    assert "no_remote_or_native_start" not in projected[
        "energy_not_started_categories"
    ]


def test_remote_start_counter_excludes_ssh_authentication_failure() -> None:
    from types import SimpleNamespace

    denied = SimpleNamespace(
        returncode=255, stdout="",
        stderr="Permission denied (publickey)",
    )
    runtime_failure = SimpleNamespace(
        returncode=2,
        stdout="[producer-e2e-eval] starting model=resnet50",
        stderr="runtime failed",
    )
    assert _native_remote_launch_observed(denied) is False
    assert _native_remote_launch_observed(runtime_failure) is True
    assert _native_global_infrastructure_failure(
        denied.stderr,
    ) == "remote_authentication_failed"
    assert _native_global_infrastructure_failure(
        "benchmark_set_invalid:yolov7_ultralytics",
    ) == ""


def test_energy_not_requested_has_no_not_started_diagnostic() -> None:
    assert _native_energy_not_started_projection(
        requested=False,
        state={"status": "not_applicable", "started_measurement_count": 0},
    ) == {}


def test_legacy_ambiguous_energy_enable_is_a_visible_generic_blocker() -> None:
    from onnx_splitpoint_tool.energy.config import (
        resolve_effective_energy_config,
    )
    from onnx_splitpoint_tool.workflow.execution_binding import (
        _energy_enabled_for_profile,
    )

    profile = {"energy": {"enabled": True}}
    effective = resolve_effective_energy_config(profile)
    assert effective["generic_energy_enabled"] is False
    assert effective["measurement_path"] == "generic"
    assert effective["configuration_errors"] == [
        "generic_energy_path_requested_but_generic_energy_disabled"
    ]
    assert _energy_enabled_for_profile(object(), profile) is False


def test_successful_plan_mode_does_not_claim_measurement_failed_to_start() -> None:
    assert _native_energy_not_started_projection(
        requested=True,
        state={
            "mode": "plan",
            "status": "ok",
            "started_measurement_count": 0,
        },
    ) == {}


def test_collector_configuration_zero_start_is_categorized() -> None:
    projected = _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "failed",
            "started_measurement_count": 0,
            "error": (
                "u.RECS collector initialization failed: calibration "
                "manifest missing"
            ),
        },
    )
    assert "collector_initialization" in projected[
        "energy_not_started_categories"
    ]
    assert "global_energy_configuration" in projected[
        "energy_not_started_categories"
    ]
    assert projected["energy_not_started_reason"].startswith(
        "Energy requested, not started: Native preflight blocked by "
    )


def test_global_remote_infrastructure_zero_start_is_categorized() -> None:
    projected = _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "failed",
            "started_measurement_count": 0,
            "stderr_tail": "ssh: Permission denied (publickey)",
        },
    )
    assert "global_remote_infrastructure" in projected[
        "energy_not_started_categories"
    ]


def test_platform_lock_zero_start_is_categorized() -> None:
    projected = _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "failed",
            "started_measurement_count": 0,
            "error": "platform_lock unavailable",
        },
    )
    assert "platform_lock" in projected["energy_not_started_categories"]


def test_quality_error_with_no_runnable_row_is_categorized() -> None:
    projected = _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "blocked_no_runtime_constructible_rows",
            "started_measurement_count": 0,
            "blocked_reason": "preprocessing_contract missing",
        },
    )
    assert "technical_quality_no_runtime_rows" in projected[
        "energy_not_started_categories"
    ]
    assert "native_runtime_preflight" in projected[
        "energy_not_started_categories"
    ]


def test_no_runtime_rows_without_quality_error_is_not_mislabeled_quality() -> None:
    projected = _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "blocked_no_runtime_constructible_rows",
            "started_measurement_count": 0,
            "started_remote_count": 0,
            "started_performance_count": 0,
        },
    )
    assert "technical_quality_no_runtime_rows" not in projected[
        "energy_not_started_categories"
    ]
    assert "native_runtime_preflight" in projected[
        "energy_not_started_categories"
    ]
    assert "no_remote_or_native_start" in projected[
        "energy_not_started_categories"
    ]


def test_started_raw_energy_does_not_get_not_started_diagnostic() -> None:
    assert _native_energy_not_started_projection(
        requested=True,
        state={
            "status": "failed_measurement_or_aggregate_contract",
            "started_measurement_count": 1,
        },
    ) == {}


def test_unverified_admission_clamps_all_quality_labels_to_raw() -> None:
    from onnx_splitpoint_tool.native_energy_reporting import (
        _final_energy_quality_result_fields,
    )

    assert _final_energy_quality_result_fields(
        plan_qualified=True,
        admission_verified=False,
        measurement_started=True,
        raw_energy_collected=True,
    ) == {
        "energy_quality_qualified": False,
        "energy_quality_status": "raw_energy_quality_not_qualified",
        "native_energy_after_technical_error": (
            "collect_raw_quality_unqualified"
        ),
    }


def test_gui_extracts_concrete_energy_not_started_reason() -> None:
    from onnx_splitpoint_tool.gui.app import (
        _evaluation_energy_not_started_messages,
    )

    message = (
        "Energy requested, not started: Native preflight blocked by "
        "collector_initialization: u.RECS unavailable"
    )
    payload = {
        "stage_results": [{
            "name": "run_native_producers",
            "details": {"energy_not_started_reason": message},
        }],
    }
    assert _evaluation_energy_not_started_messages(payload) == [message]


def test_gui_extracts_model_local_energy_skip_alongside_measured_rows() -> None:
    from onnx_splitpoint_tool.gui.app import (
        _evaluation_energy_not_started_messages,
    )

    message = (
        "Energy requested, not started: Native preflight blocked by "
        "benchmark_set_invalid:yolov7_ultralytics"
    )
    payload = {
        "stage_results": [{
            "name": "run_native_producers",
            "details": {
                "started_measurement_count": 3,
                "model_preflight_rows": [{
                    "model": "yolov7_ultralytics",
                    "energy_not_started_reason": message,
                }],
            },
        }],
    }
    assert _evaluation_energy_not_started_messages(payload) == [message]
