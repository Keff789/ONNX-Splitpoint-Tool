from __future__ import annotations

from pathlib import Path

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan


ROOT = Path(__file__).resolve().parents[1]
PROFILE = (
    ROOT / "profiles" / "resnet50_v27539_deepx_full_quality_canary.yaml"
)
SETUP_ID = "orin_nx_deepx_m1_01"


def test_packaged_v27539_deepx_canary_raw_schema_and_load_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = yaml.safe_load(PROFILE.read_text(encoding="utf-8"))

    assert source["name"] == "resnet50_v27539_deepx_full_quality_canary"
    assert "v2.75.39" in source["purpose"]
    assert validate_evaluation_profile_payload(
        source,
        source_path=PROFILE,
    ) == source
    assert source["execution_preset"]["id"] == "standard"
    assert source["execution_preset"]["follow_tool_config"] is False
    assert source["execution_preset"]["overrides"] == {
        "native_enabled": False,
        "energy_enabled": False,
    }
    assert (
        source["execution_preset"]["snapshot"]["build"]["deepx"][
            "force_build"
        ]
        is False
    )
    assert [row["id"] for row in source["run_profiles"]] == [
        "ort_tensorrt",
        "deepx_m1_full",
    ]

    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "missing_user_run_modes.yaml"),
    )
    loaded = load_evaluation_profile(PROFILE, validate=True)

    assert loaded is not None
    assert loaded.profile_id == source["name"]
    assert loaded.source_profile == source
    assert loaded.start_snapshot is not None
    assert loaded.start_snapshot["consistency"] == {
        "status": "ok",
        "mismatches": [],
    }
    profile = loaded.raw_profile
    assert profile["deepx_build"]["force_build"] is False
    assert profile["deepx_build"]["calib_count"] == 500
    assert profile["deepx_build"]["calibration_method"] == "ema"
    assert profile["deepx_build"]["opt_level"] == 0


def test_packaged_v27539_deepx_canary_has_exact_effective_plan(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "missing_user_run_modes.yaml"),
    )
    loaded = load_evaluation_profile(PROFILE, validate=True)
    assert loaded is not None
    profile = loaded.raw_profile
    plan = build_effective_execution_plan(profile)

    canary = plan["quality_canary"]
    assert canary["enabled"] is True
    assert canary["ok"] is True
    assert canary["status"] == "ready"
    assert canary["errors"] == []
    assert plan["quality_canary_enabled"] is True
    assert plan["quality_canary_execution_scope"] == "full_only"

    assert plan["models"] == ["resnet50"]
    assert plan["generic_rows_total"] == 0
    assert plan["expected_generic_result_rows_total"] == 0
    assert plan["effective_generic_run_ids"] == []
    assert plan["expected_full_quality_results_per_model"] == 2
    assert plan["expected_full_quality_results_total"] == 2
    assert plan["remote_run_invocations_per_model"] == 1
    assert plan["remote_run_invocations_total"] == 1
    assert plan["setup_groups"] == {
        SETUP_ID: ["deepx_m1_full", "ort_tensorrt"],
    }
    assert [
        (row["id"], row["backend"], row["dispatch_run_id"])
        for row in plan["expected_full_quality_identities"]
    ] == [
        ("deepx_m1_full", "deepx_m1", "deepx_m1_full"),
        ("tensorrt_at_deepx_m1_full", "tensorrt", "ort_tensorrt"),
    ]

    assert plan["native_enabled"] is False
    assert plan["native_full_baselines"] is False
    assert plan["native_split_backends"] == []
    assert plan["native_energy_enabled"] is False
    assert plan["generic_energy_enabled"] is False
    assert plan["energy_measurement_path"] == "disabled"
    assert plan["ranking_enabled"] is False
    assert plan["performance_claims_emitted"] is False
    assert plan["tensorrt_performance_owner_group"] == ""
    assert all(
        row["performance_claims_emitted"] is False
        for row in plan["expected_full_quality_identities"]
    )

    assert plan["calibration_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert plan["validation_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert plan["bootstrap_repetitions"] == 500
    assert profile["deepx_build"]["force_build"] is False
