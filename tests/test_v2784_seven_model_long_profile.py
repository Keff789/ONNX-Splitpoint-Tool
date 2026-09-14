from __future__ import annotations

from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.benchmark.services import (
    _hailo_pair_parallel_decision_v27550,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/complete_set_7models_v2784_b500_audit20.yaml"
MODEL_IDS = [
    "resnet50",
    "yolo26s",
    "yolov7_paper",
    "mobilenet_v3_large",
    "regnet_x_1_6gf",
    "yolo26m",
    "yolo11l",
]
HARDWARE_SETUPS = (
    "/home/kmika/.onnx_splitpoint_tool/frozen/"
    "v2784_7model_hardware_setups.yaml"
)


def _source() -> dict:
    payload = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def test_v2784_long_profile_freezes_scientific_scope_and_b500_contract() -> None:
    source = _source()
    note = source["implementation_note"]
    assert "Protocol deviation accepted by the user on 2026-08-28" in note
    assert "replaces a separate B5 canary" in note
    assert "does not relax evidence, quality, or candidate-universe" in note

    models = source["model_suite"]["primary"]
    assert [row["id"] for row in models] == MODEL_IDS
    assert source["model_suite"]["reserve"] == []
    assert "yolov7" not in MODEL_IDS
    assert "yolo26x" not in MODEL_IDS
    assert [row["onnx"] for row in models] == [
        f"/home/kmika/Models/{model_id}.onnx" for model_id in MODEL_IDS
    ]
    assert {
        row["id"]: (row["evaluation_role"], row["generalization_scope"])
        for row in models
    } == {
        "resnet50": ("development", "development"),
        "yolo26s": ("development", "development"),
        "yolov7_paper": ("development", "development"),
        "mobilenet_v3_large": (
            "confirmatory_holdout", "model_family_holdout",
        ),
        "regnet_x_1_6gf": (
            "confirmatory_holdout", "model_family_holdout",
        ),
        "yolo26m": ("confirmatory_holdout", "within_family_transfer"),
        "yolo11l": ("confirmatory_holdout", "model_family_holdout"),
    }
    assert all(
        row["candidate_universe"] == {"mode": "deterministic_audit"}
        for row in models
    )

    selection = source["selection_policy"]
    assert selection["selection_strategy"] == "score_independent_audit"
    assert selection["score_independent_audit_enabled"] is True
    assert selection["audit_candidate_universe"] == "deterministic_audit"
    assert selection["audit_size"] == 20
    assert selection["minimum_valid_audit_candidates"] == 10
    assert selection["audit_seed"] == 20260710
    assert selection["min_gap"] == 3
    assert selection["candidate_search_pool"] == 60
    assert selection["require_single_part2_input"] is False
    # This limit applies only to the deployment shortlist.  It does not reduce
    # the independent twenty-candidate formal audit below.
    assert selection["max_accepted_cases_per_model"] == 1
    assert selection["forced_cases"] == {"yolo11l": ["b067"]}
    assert "seed_cases" not in selection

    ranking = source["ranking_validation"]
    assert ranking["methods"] == ["cut_bytes_only"]
    assert ranking["candidate_universe"] == "deterministic_audit"
    assert ranking["audit_size"] == 20
    assert ranking["minimum_valid_audit_candidates"] == 10
    assert ranking["audit_seed"] == 20260710
    assert ranking["require_frozen_predictions"] is True
    assert ranking["require_complete_candidate_universe"] is True

    assert source["validation_execution"]["max_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert source["quality_gate"]["statistics"] == {
        "method": "paired_bootstrap",
        "confidence_level": 0.95,
        "bootstrap_repetitions": 500,
        "seed": 20260710,
        "decision": "lower_one_sided_bound",
        "execution_location": "central_management",
        "workers": 4,
    }
    hailo = source["hailo_build"]
    assert hailo["targets"] == ["hailo8", "hailo10"]
    assert hailo["optimization_level"] == 1
    assert hailo["calib_count"] == 500
    assert hailo["calib_batch_size"] == 8
    assert hailo["calibration_storage"] == "memmap"
    assert hailo["cache_enabled"] is True
    assert hailo["force_build"] is False
    deepx = source["deepx_build"]
    assert deepx == {
        "mode": "reuse_and_build_missing",
        "target": "deepx_m1",
        "calib_dir": "",
        "calib_count": 500,
        "calibration_method": "ema",
        "opt_level": 0,
        "force_build": False,
        "classification_preprocessing": "imagenet_mean_std",
    }
    assert source["official_coco_evaluation"]["enabled"] is False
    assert source["native_producers"]["enabled"] is False
    assert source["native_producers"]["split_backends"] == []
    assert source["energy"]["enabled"] is False
    assert source["energy"]["generic_enabled"] is False


def test_v2784_long_profile_materializes_frozen_hardware_and_audit_union(
    monkeypatch,
) -> None:
    loaded = load_evaluation_profile(
        str(PROFILE), base_dir=PROFILE.parent, validate=True,
    )
    assert loaded is not None
    assert loaded.profile_id == "complete_set_7models_v2784_b500_audit20"
    profile = loaded.raw_profile

    assert profile["hardware"] == {
        "setups_file": HARDWARE_SETUPS,
        "selected_setups": [],
        "selected_groups": [],
    }
    assert profile["build_scheduler"] == {
        "enabled": True,
        "max_workers": 3,
        "cpu_tokens": 8,
        "ram_mb": 0,
        "ram_reserve_mb": 2048,
        "family_limits": {"hailo8": 1, "hailo10": 1, "deepx": 1},
        "weights": {
            "hailo8": {"cpu_tokens": 4, "ram_mb": 6144},
            "hailo10": {"cpu_tokens": 4, "ram_mb": 6144},
            "deepx": {"cpu_tokens": 2, "ram_mb": 4096},
        },
        "prefetch_deepx_full": True,
        "pipeline_next_model": False,
    }

    monkeypatch.setattr(
        "onnx_splitpoint_tool.execution_plan.matrix_for_runtime",
        lambda _profile: [],
    )
    plan = build_effective_execution_plan(profile)
    twenty = {model_id: 20 for model_id in MODEL_IDS}
    ten = {model_id: 10 for model_id in MODEL_IDS}
    twenty_one = {model_id: 21 for model_id in MODEL_IDS}
    one = {model_id: 1 for model_id in MODEL_IDS}

    assert plan["models"] == MODEL_IDS
    assert plan["score_independent_audit_counts"] == twenty
    assert plan["score_independent_audit_minimum_valid_counts"] == ten
    assert plan["candidate_counts_by_model"] == twenty
    assert plan["deployment_shortlist_upper_bound_by_model"] == one
    assert plan["execution_union_candidate_counts_min_by_model"] == twenty
    assert (
        plan["execution_union_candidate_counts_upper_bound_by_model"]
        == twenty_one
    )
    assert plan["logical_run_profiles"] == [
        "hailo8",
        "hailo8_to_trt",
        "hailo10",
        "hailo10_to_tensorrt",
        "deepx_m1_full",
        "deepx_m1_to_tensorrt",
    ]
    assert plan["expected_generic_result_rows_min_by_model"] == {
        model_id: 84 for model_id in MODEL_IDS
    }
    assert plan["expected_generic_result_rows_by_model"] == {
        model_id: 88 for model_id in MODEL_IDS
    }
    assert plan["expected_generic_result_rows_min_total"] == 588
    assert plan["expected_generic_result_rows_total"] == 616
    assert plan["generic_runtime_enabled"] is True
    assert plan["native_split_backends"] == []
    assert plan["energy_measurement_path"] == "disabled"
    assert plan["validation_items"] == {
        "classification": 500,
        "detection": 500,
    }


def test_v2784_hailo_pair_parallelism_is_resource_gated(monkeypatch) -> None:
    scheduler = _source()["build_scheduler"]
    assert scheduler["cpu_tokens"] == 8
    assert scheduler["ram_mb"] == 0
    assert scheduler["ram_reserve_mb"] == 2048

    monkeypatch.setattr(
        "onnx_splitpoint_tool.benchmark.services._available_ram_mb_v27550",
        lambda: 14336,
    )
    parallel = _hailo_pair_parallel_decision_v27550(
        ["hailo8", "hailo10"], scheduler, backend="venv",
    )
    assert parallel["effective"] is True
    assert parallel["reason"] == "resources_available"
    assert parallel["cpu_required"] == 8
    assert parallel["ram_required_mb"] == 12288
    assert parallel["ram_pool_mb"] == 12288

    monkeypatch.setattr(
        "onnx_splitpoint_tool.benchmark.services._available_ram_mb_v27550",
        lambda: 14000,
    )
    serial = _hailo_pair_parallel_decision_v27550(
        ["hailo8", "hailo10"], scheduler, backend="venv",
    )
    assert serial["effective"] is False
    assert serial["reason"].startswith("insufficient_ram_mb:")

