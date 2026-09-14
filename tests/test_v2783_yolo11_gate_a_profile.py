from __future__ import annotations

from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.benchmark.services import (
    normalize_hailo_feasibility_control,
)
from onnx_splitpoint_tool.execution_plan import (
    build_effective_execution_plan,
)
from onnx_splitpoint_tool.run_modes import infer_run_mode


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml"
SOURCE_B5_SHA256 = (
    "13f486b0e53e335dfb4407bc09f364a116a3cfc774757e3a46bfd3c7c86da91f"
)
EVIDENCE_INDEX = (
    "/home/kmika/.onnx_splitpoint_tool/build_evidence/"
    "v2783_yolo11_gate_a.json"
)
EVIDENCE_ARTIFACT_ROOT = (
    "/home/kmika/Models/EvaluationRuns/"
    "complete_set_v2782_canary_b5_20260827_193658"
)


def test_yolo11_gate_a_profile_is_exact_and_loads_effective_plan(
    monkeypatch,
) -> None:
    profile_source = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert profile_source["implementation_note"].find(SOURCE_B5_SHA256) >= 0
    assert profile_source["selection_policy"] == {
        "max_accepted_cases_per_model": 1,
        "preferred_shortlist": 3,
        "min_gap": 3,
        "candidate_search_pool": 10,
        "require_single_part2_input": False,
        "selection_strategy": "stratified_windows",
        "score_independent_audit_enabled": False,
        "audit_candidate_universe": "deterministic_audit",
        "audit_size": 20,
        "minimum_valid_audit_candidates": 1,
        "audit_seed": 20260710,
        "report_blocked_configs": True,
        "report_plan_adjustments": True,
        "keep_partial_hailo_cases": True,
        "full_model_hailo_preflight_policy": "skip",
    }
    assert profile_source["model_suite"]["primary"] == [
        {
            "id": "yolo11l",
            "task": "detection",
            "family": "yolo11",
            "source": "ultralytics",
            "semantic_dataset": "coco2017_val",
            "development_subset": "coco_50",
            "input_shape": [1, 3, 640, 640],
            "onnx": "/home/kmika/Models/yolo11l.onnx",
            "family_id": "yolo11",
            "evaluation_role": "development",
            "generalization_scope": "development",
            "validation_tier": "screening",
            "candidate_universe": {"mode": "declared_shortlist"},
        }
    ]

    loaded = load_evaluation_profile(
        str(PROFILE), base_dir=PROFILE.parent, validate=True,
    )
    assert loaded is not None
    raw = loaded.raw_profile
    monkeypatch.setattr(
        "onnx_splitpoint_tool.execution_plan.matrix_for_runtime",
        lambda _profile: [],
    )
    plan = build_effective_execution_plan(raw)

    assert loaded.profile_id == "yolo11l_v2783_hailo8_first_b5_gate_a"
    assert infer_run_mode(raw) == "standard"
    assert [row["id"] for row in raw["model_suite"]["primary"]] == [
        "yolo11l"
    ]
    assert plan["models"] == ["yolo11l"]
    assert plan["candidate_counts_by_model"] == {"yolo11l": 1}
    assert plan["logical_run_profiles"] == [
        "hailo8_to_trt", "hailo10_to_tensorrt",
    ]
    assert plan["generic_rows_total"] == 0
    assert plan["generic_runtime_enabled"] is False

    hailo = raw["hailo_build"]
    assert hailo["targets"] == ["hailo8", "hailo10"]
    assert hailo["build_full"] is False
    assert hailo["build_part1"] is True
    assert hailo["build_part2"] is False
    assert hailo["calib_count"] == 500
    assert hailo["calib_batch_size"] == 8
    assert hailo["optimization_level"] == 1
    assert hailo["cache_enabled"] is True
    assert hailo["force_build"] is False

    control = normalize_hailo_feasibility_control(
        hailo["feasibility_control"]
    )
    assert control == {
        "enabled": True,
        "mode": "hailo8_first_common_anchor",
        "primary_target": "hailo8",
        "gated_target": "hailo10h",
        "required_variant": "part1",
        "max_hailo8_cold_attempts": 3,
        "max_total_cold_builds": 4,
        "max_cold_builds_per_boundary": 2,
        "wall_time_budget_s": 21600,
        "parser_timeout_s": 600,
        "allow_recipe_retries": False,
        "stop_workflow_on_exhaustion": True,
        "evidence_index_path": EVIDENCE_INDEX,
        "evidence_artifact_root": EVIDENCE_ARTIFACT_ROOT,
    }

    assert raw["workflow"]["execution_mode"] == "generate_benchmarksets"
    assert raw["workflow"]["skip_runtime_benchmarks"] is True
    assert raw["workflow"]["stop_after"] == "build_backend_artifacts"
    assert raw["remote_execution"]["enabled"] is False
    assert raw["native_producers"]["enabled"] is False
    assert raw["energy"]["enabled"] is False
    assert raw["ranking_validation"]["enabled"] is False
    assert raw["validation_execution"]["max_items"] == {"detection": 5}
