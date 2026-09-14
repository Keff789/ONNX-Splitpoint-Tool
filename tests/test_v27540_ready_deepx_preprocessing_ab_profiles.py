from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
    validate_evaluation_profile_payload,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan


ROOT = Path(__file__).resolve().parents[1]
PROFILE_A = (
    ROOT
    / "profiles"
    / "resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml"
)
PROFILE_B = (
    ROOT
    / "profiles"
    / "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml"
)
SETUP_ID = "orin_nx_deepx_m1_01"
CALIBRATION_MANIFEST = (
    "/home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/"
    "imagenet_train_calibration_manifest.json"
)
VALIDATION_MANIFEST = (
    "/home/kmika/.onnx_splitpoint_tool/final_datasets/manifests/"
    "imagenet_val_manifest.json"
)
EXPECTED_ARMS = {
    PROFILE_A: (
        "resnet50_v27540_deepx_preprocess_a_current_scale_only",
        "current_scale_only",
        "~/Models/BackendArtifacts/deepx/v2.75.40/"
        "resnet50_preprocessing_ab/a_current_scale_only",
    ),
    PROFILE_B: (
        "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std",
        "imagenet_mean_std",
        "~/Models/BackendArtifacts/deepx/v2.75.40/"
        "resnet50_preprocessing_ab/b_imagenet_mean_std",
    ),
}


def _leaf_differences(
    left: Any, right: Any, *, prefix: tuple[str, ...] = (),
) -> set[tuple[str, ...]]:
    if isinstance(left, dict) and isinstance(right, dict):
        differences: set[tuple[str, ...]] = set()
        for key in sorted(set(left) | set(right)):
            path = (*prefix, str(key))
            if key not in left or key not in right:
                differences.add(path)
            else:
                differences.update(
                    _leaf_differences(left[key], right[key], prefix=path)
                )
        return differences
    if isinstance(left, list) and isinstance(right, list):
        differences = set()
        for index in range(max(len(left), len(right))):
            path = (*prefix, str(index))
            if index >= len(left) or index >= len(right):
                differences.add(path)
            else:
                differences.update(
                    _leaf_differences(
                        left[index], right[index], prefix=path,
                    )
                )
        return differences
    return set() if left == right else {prefix}


def test_packaged_v27540_ab_raw_profiles_have_only_the_arm_differences() -> None:
    raw_a = yaml.safe_load(PROFILE_A.read_text(encoding="utf-8"))
    raw_b = yaml.safe_load(PROFILE_B.read_text(encoding="utf-8"))

    assert validate_evaluation_profile_payload(
        raw_a, source_path=PROFILE_A,
    ) == raw_a
    assert validate_evaluation_profile_payload(
        raw_b, source_path=PROFILE_B,
    ) == raw_b
    assert _leaf_differences(raw_a, raw_b) == {
        ("name",),
        (
            "execution_preset", "snapshot", "build", "deepx",
            "classification_preprocessing",
        ),
        (
            "execution_preset", "snapshot", "build", "deepx",
            "cache_dir",
        ),
    }

    for path, (profile_id, mode, cache_dir) in EXPECTED_ARMS.items():
        source = raw_a if path == PROFILE_A else raw_b
        assert source["name"] == profile_id
        assert source["execution_preset"]["id"] == "standard"
        assert source["execution_preset"]["follow_tool_config"] is False
        assert source["execution_preset"]["overrides"] == {
            "native_enabled": False,
            "energy_enabled": False,
        }
        deepx = source["execution_preset"]["snapshot"]["build"]["deepx"]
        assert deepx == {
            "mode": "reuse_and_build_missing",
            "optimization_level": 0,
            "calibration_items": 500,
            "calibration_method": "ema",
            "classification_preprocessing": mode,
            "cache_dir": cache_dir,
            "force_build": False,
        }
        manifests = source["campaign"]["dataset_manifests"]
        assert manifests["classification"] == {
            "calibration": CALIBRATION_MANIFEST,
            "validation": VALIDATION_MANIFEST,
        }


def test_packaged_v27540_ab_load_and_effective_plan_contract(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "missing_user_run_modes.yaml"),
    )

    for path, (profile_id, mode, cache_dir) in EXPECTED_ARMS.items():
        loaded = load_evaluation_profile(path, validate=True)
        assert loaded is not None
        assert loaded.profile_id == profile_id
        assert loaded.start_snapshot is not None
        assert loaded.start_snapshot["consistency"] == {
            "status": "ok",
            "mismatches": [],
        }
        profile = loaded.raw_profile
        assert profile["campaign"]["dataset_manifests"]["classification"] == {
            "calibration": CALIBRATION_MANIFEST,
            "validation": VALIDATION_MANIFEST,
        }
        assert profile["deepx_build"] == {
            "mode": "reuse_and_build_missing",
            "target": "deepx_m1",
            "calib_dir": "",
            "calib_count": 500,
            "calibration_method": "ema",
            "opt_level": 0,
            "force_build": False,
            "classification_preprocessing": mode,
            "cache_dir": cache_dir,
        }

        plan = build_effective_execution_plan(profile)
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
        assert plan["quality_canary_enabled"] is True
        assert plan["quality_canary_execution_scope"] == "full_only"
        assert plan["native_enabled"] is False
        assert plan["native_full_baselines"] is False
        assert plan["native_energy_enabled"] is False
        assert plan["generic_energy_enabled"] is False
        assert plan["energy_measurement_path"] == "disabled"
        assert plan["ranking_enabled"] is False
        assert plan["performance_claims_emitted"] is False
        assert plan["calibration_items"] == {
            "classification": 500,
            "detection": 500,
        }
        assert plan["validation_items"] == {
            "classification": 500,
            "detection": 500,
        }
        assert plan["bootstrap_repetitions"] == 500
        assert plan["deepx_classification_preprocessing"] == mode
        assert plan["deepx_classification_preprocessing_explicit"] is True
        assert plan["deepx_full_cache_contract"] == "v2_exact_explicit"
        assert plan["deepx_cache_dir"] == cache_dir
