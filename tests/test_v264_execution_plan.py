from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool.execution_plan import (
    build_effective_execution_plan,
    execution_plan_text,
)


def _resolved_smoke_profile() -> dict:
    return {
        "model_suite": {
            "primary": [
                {"id": "resnet50", "enabled": True, "task": "classification"},
                {"id": "yolo26s", "enabled": True, "task": "detection"},
            ]
        },
        "selection_policy": {"max_accepted_cases_per_model": 1},
        "run_profiles": [
            {"id": "ort_tensorrt", "enabled": True},
            {"id": "hailo8", "enabled": True},
            {"id": "hailo8_to_trt", "enabled": True},
            {"id": "hailo10", "enabled": True},
            {"id": "hailo10_to_tensorrt", "enabled": True},
            {"id": "deepx_m1_full", "enabled": True},
            {"id": "deepx_m1_to_tensorrt", "enabled": True},
        ],
        "quality_gate": {
            "statistics": {
                "execution_location": "central_management",
                "workers": 4,
            }
        },
        "native_producers": {
            "enabled": True,
            "full_baselines": {
                "enabled": True,
                "backends": ["deepx", "hailo10h", "hailo8", "tensorrt"],
            },
        },
        "execution_preset": {
            "id": "smoke",
            "label": "Smoke",
            "overrides": {"native_enabled": True, "energy_enabled": True},
            "effective": {"native_full_baselines_enabled": True},
            "snapshot": {
                "defaults": {"native_enabled": False, "energy_enabled": False},
                # The raw mode default used to leak into the preview.
                "runtime": {
                    "native": {"full_baselines": False, "frames": 100, "warmup": 10},
                    "benchmark": {"warmup": 1, "runs": 1},
                },
                "quality": {
                    "bootstrap_repetitions": 25,
                    "execution_location": "local",
                    "workers": 1,
                },
            },
        },
    }


def test_preview_uses_resolved_native_full_and_management_quality() -> None:
    plan = build_effective_execution_plan(_resolved_smoke_profile())

    assert plan["native_full_baselines"] is True
    assert plan["quality_execution_location"] == "central_management"
    assert plan["quality_workers"] == 4
    assert plan["management_reference_profiles"] == ["ort_cpu"]
    assert "ort_cpu" in plan["automatic_reference_profiles"]
    assert "ort_cpu" not in plan["effective_generic_run_ids"]
    assert plan["generic_logical_runs_per_model"] == 7
    assert plan["generic_rows_per_model"] == 8
    assert plan["generic_rows_total"] == 16
    assert plan["remote_run_invocations_per_model"] == 3
    assert "gpu_reference_setup" not in plan["setup_groups"]
    assert "ort_tensorrt" in plan["setup_groups"]["deepx_setup"]

    rendered = execution_plan_text(plan)
    assert "full baselines=on" in rendered
    assert "Quality reference: central_management · workers=4" in rendered


def test_local_quality_keeps_remote_cpu_reference() -> None:
    profile = _resolved_smoke_profile()
    profile["quality_gate"]["statistics"] = {
        "execution_location": "local",
        "workers": 1,
    }
    plan = build_effective_execution_plan(profile)

    assert plan["quality_execution_location"] == "local"
    assert plan["management_reference_profiles"] == []
    assert "ort_cpu" in plan["effective_generic_run_ids"]


def test_gui_defaults_no_longer_revert_smoke_to_local_quality() -> None:
    profile_editor = Path("onnx_splitpoint_tool/gui/profile_editor.py").read_text(encoding="utf-8")
    run_mode_editor = Path("onnx_splitpoint_tool/gui/run_mode_editor.py").read_text(encoding="utf-8")

    assert '("local" if mode_id == "smoke" else "central_management")' not in profile_editor
    assert 'value="central_management"' in profile_editor
    assert 'self.var_quality_workers = tk.IntVar(self, value=4)' in profile_editor
    assert "Smoke remains local" not in run_mode_editor


def test_hailo_only_plan_places_automatic_tensorrt_reference_on_existing_setup() -> None:
    profile = _resolved_smoke_profile()
    profile["model_suite"]["primary"] = profile["model_suite"]["primary"][:1]
    profile["run_profiles"] = [{"id": "hailo8", "enabled": True}]
    plan = build_effective_execution_plan(profile)

    assert plan["setup_groups"] == {
        "hailo8_setup": ["hailo8", "ort_tensorrt"],
    }
    assert plan["remote_run_invocations_per_model"] == 1
    assert plan["cold_suite_uploads_per_model"] == 1


def test_reference_first_profile_still_uses_the_first_physical_setup() -> None:
    profile = _resolved_smoke_profile()
    profile["model_suite"]["primary"] = profile["model_suite"]["primary"][:1]
    profile["run_profiles"] = [
        {"id": "ort_tensorrt", "enabled": True},
        {"id": "hailo10", "enabled": True},
    ]
    plan = build_effective_execution_plan(profile)

    assert plan["setup_groups"] == {
        "hailo10h_setup": ["hailo10", "ort_tensorrt"],
    }
    assert plan["remote_run_invocations_per_model"] == 1


def test_development_score_independent_audit_sizes_execution_union() -> None:
    profile = _resolved_smoke_profile()
    profile["model_suite"]["primary"] = [
        {
            "id": "resnet50",
            "enabled": True,
            "task": "classification",
            "evaluation_role": "development",
        }
    ]
    profile["selection_policy"].update({
        "max_accepted_cases_per_model": 1,
        "selection_strategy": "score_independent_audit",
        "audit_size": 4,
        "minimum_valid_audit_candidates": 3,
    })

    plan = build_effective_execution_plan(profile)

    assert plan["score_independent_audit_enabled"] is True
    assert plan["score_independent_audit_requested"] is True
    assert plan["score_independent_audit_counts"] == {"resnet50": 4}
    assert plan["development_audit_counts"] == {"resnet50": 4}
    assert plan["holdout_audit_counts"] == {}
    assert plan["score_independent_audit_minimum_valid_counts"] == {
        "resnet50": 3,
    }
    assert plan["candidate_counts_by_model"] == {"resnet50": 4}
    assert plan["deployment_shortlist_upper_bound_by_model"] == {
        "resnet50": 1,
    }
    assert plan["execution_union_candidate_counts_min_by_model"] == {
        "resnet50": 4,
    }
    assert plan["execution_union_candidate_counts_upper_bound_by_model"] == {
        "resnet50": 5,
    }
    # Seven effective Generic run IDs produce 20 rows for four candidates and
    # at most 24 when the one deployment candidate does not overlap the audit.
    assert plan["expected_generic_result_rows_min_by_model"] == {
        "resnet50": 20,
    }
    assert plan["expected_generic_result_rows_by_model"] == {
        "resnet50": 24,
    }
    assert plan["expected_generic_result_rows_min_total"] == 20
    assert plan["expected_generic_result_rows_total"] == 24
    assert plan["generic_rows_total"] == 24

    rendered = execution_plan_text(plan)
    assert "Score-independent audit candidates: {'resnet50': 4}" in rendered
    assert "Execution union candidates/model: {'resnet50': '4–5'}" in rendered
    assert "Generic normalized result rows (planned): 20–24" in rendered
    assert "Hold-out audit candidates" not in rendered


def test_non_audit_execution_plan_remains_exact() -> None:
    profile = _resolved_smoke_profile()
    profile["model_suite"]["primary"] = profile["model_suite"]["primary"][:1]

    plan = build_effective_execution_plan(profile)

    assert plan["score_independent_audit_enabled"] is False
    assert plan["score_independent_audit_counts"] == {}
    assert plan["development_audit_counts"] == {}
    assert plan["holdout_audit_counts"] == {}
    assert plan["candidate_counts_by_model"] == {"resnet50": 1}
    assert plan["execution_union_candidate_counts_min_by_model"] == {
        "resnet50": 1,
    }
    assert plan["execution_union_candidate_counts_upper_bound_by_model"] == {
        "resnet50": 1,
    }
    assert plan["expected_generic_result_rows_min_total"] == 8
    assert plan["expected_generic_result_rows_total"] == 8

    rendered = execution_plan_text(plan)
    assert "Score-independent audit candidates: none" in rendered
    assert "Execution union candidates/model: {'resnet50': '1'}" in rendered
    assert "Generic normalized result rows (planned): 8 " in rendered


def test_explicit_audit_enable_flag_activates_audit_with_legacy_strategy() -> None:
    profile = _resolved_smoke_profile()
    profile["model_suite"]["primary"] = profile["model_suite"]["primary"][:1]
    profile["selection_policy"].update({
        "selection_strategy": "stratified_windows",
        "score_independent_audit_enabled": True,
        "audit_size": 4,
        "minimum_valid_audit_candidates": 3,
    })

    plan = build_effective_execution_plan(profile)

    assert plan["score_independent_audit_requested"] is True
    assert plan["score_independent_audit_counts"] == {"resnet50": 4}
    assert plan["execution_union_candidate_counts_upper_bound_by_model"] == {
        "resnet50": 5,
    }
