from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    load_evaluation_profile,
)
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan


ROOT = Path(__file__).resolve().parents[1]
PROFILE = ROOT / "profiles/yolov7_paper_v27547_standard_anchor_b500.yaml"
MODEL_SHA256 = (
    "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d"
)


class _UniqueKeySafeLoader(yaml.SafeLoader):
    pass


def _construct_unique_mapping(loader, node, deep=False):
    mapping = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if key in mapping:
            raise yaml.constructor.ConstructorError(
                "while constructing a mapping",
                node.start_mark,
                f"duplicate key: {key!r}",
                key_node.start_mark,
            )
        mapping[key] = loader.construct_object(value_node, deep=deep)
    return mapping


_UniqueKeySafeLoader.add_constructor(
    yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
    _construct_unique_mapping,
)


def test_v27547_shipped_profile_contains_no_duplicate_yaml_keys() -> None:
    payload = yaml.load(
        PROFILE.read_text(encoding="utf-8"),
        Loader=_UniqueKeySafeLoader,
    )
    assert payload["execution_preset"]["snapshot"]["holdout"]["seed"] == 20260710
    assert payload["model_suite"]["primary"][0]["candidate_universe"]["seed"] == 20260710


def test_unique_key_loader_rejects_a_real_duplicate() -> None:
    with pytest.raises(yaml.constructor.ConstructorError, match="duplicate key"):
        yaml.load("holdout:\n  seed: 1\n  seed: 2\n", Loader=_UniqueKeySafeLoader)


def test_v27547_yolov7_anchor_effective_plan_is_exact(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "run_modes.yaml"),
    )
    loaded = load_evaluation_profile(
        str(PROFILE), base_dir=PROFILE.parent, validate=True,
    )
    monkeypatch.setattr(
        "onnx_splitpoint_tool.execution_plan.matrix_for_runtime",
        lambda _profile: [
            {
                "id": "orin_nx_hailo8_01",
                "accelerator": "hailo8",
                "enabled": True,
                "runtime": {"enabled": True, "host": "h8", "user": "nx"},
            },
            {
                "id": "orin_nx_hailo10_01",
                "accelerator": "hailo10h",
                "enabled": True,
                "runtime": {"enabled": True, "host": "h10", "user": "nx"},
            },
            {
                "id": "orin_nx_deepx_m1_01",
                "accelerator": "deepx_m1",
                "enabled": True,
                "runtime": {"enabled": True, "host": "dx", "user": "nx"},
            },
        ],
    )
    plan = build_effective_execution_plan(loaded.raw_profile)

    assert loaded.profile_id == "yolov7_paper_v27547_standard_anchor_b500"
    assert loaded.start_snapshot["consistency"] == {
        "status": "ok",
        "mismatches": [],
    }
    assert [row["id"] for row in loaded.raw_profile["model_suite"]["primary"]] == [
        "yolov7_paper"
    ]
    assert loaded.raw_profile["model_suite"]["primary"][0]["model_sha256"] == MODEL_SHA256
    assert loaded.raw_profile["selection_policy"]["forced_cases"] == {
        "yolov7_paper": ["b044"],
    }
    assert plan["models"] == ["yolov7_paper"]
    assert plan["candidate_counts_by_model"] == {"yolov7_paper": 1}
    assert plan["generic_rows_total"] == 8
    assert plan["generic_runtime_enabled"] is True
    assert plan["native_enabled"] is True
    assert plan["native_full_baselines"] is True
    assert plan["native_frames"] == 1000
    assert plan["native_warmup"] == 100
    assert plan["native_performance_repetitions"] == 3
    assert plan["native_split_backends"] == ["hailo8", "hailo10h", "deepx"]
    # Per setup: one selected split, one accelerator Full and one setup-local
    # TensorRT Full. Three setups by three roles by three repetitions.
    native_full_rows = len(plan["native_split_backends"]) * 3
    assert native_full_rows == 9
    assert native_full_rows * plan["native_performance_repetitions"] == 27
    companions = plan["setup_local_tensorrt_quality_companion_contract"]
    assert companions["status"] == "ready"
    assert companions["models"] == ["yolov7_paper"]
    assert companions["identity_count"] == 3
    assert companions["quality_companion_endpoint_ids"] == [
        "tensorrt_at_hailo8_full",
        "tensorrt_at_hailo10h_full",
        "tensorrt_at_deepx_m1_full",
    ]
    assert plan["generic_energy_enabled"] is False
    assert plan["native_energy_enabled"] is False
    assert plan["ranking_enabled"] is False
    assert plan["score_independent_audit_enabled"] is False
    assert plan["calibration_items"]["detection"] == 500
    assert plan["validation_items"]["detection"] == 500
    assert plan["bootstrap_repetitions"] == 500
    assert plan["official_coco_enabled"] is True
    assert plan["official_coco_required"] is True
    official = loaded.raw_profile["official_coco_evaluation"]
    assert official["enabled"] is True
    assert official["required_for_final"] is True
    assert official["require_pycocotools"] is True


def test_v27547_anchor_is_fresh_results_but_retains_normal_build_cache_policy(
    monkeypatch, tmp_path: Path,
) -> None:
    monkeypatch.setenv(
        "ONNX_SPLITPOINT_RUN_MODES_FILE",
        str(tmp_path / "run_modes.yaml"),
    )
    loaded = load_evaluation_profile(
        str(PROFILE), base_dir=PROFILE.parent, validate=True,
    )
    payload = loaded.raw_profile
    snapshot = payload["execution_preset"]["snapshot"]
    text = PROFILE.read_text(encoding="utf-8")

    assert snapshot["reproducibility"]["verify_model_content"] is True
    assert snapshot["quality"]["cache_task_quality"] is False
    assert snapshot["holdout"]["prediction_freeze_enabled"] is False
    assert snapshot["build"]["hailo"]["cache_integrity"] == "relaxed"
    assert snapshot["build"]["hailo"]["force_build"] is False
    assert snapshot["build"]["deepx"]["cache_dir"] == (
        "~/Models/BackendArtifacts/deepx/v2.75.44/"
        "thesis_standard_b500_imagenet_mean_std"
    )
    assert snapshot["build"]["artifact_store"]["verify_on_reuse"] == "metadata"
    assert snapshot["runtime"]["remote_cache"]["reuse_tensorrt_engines"] is True
    assert not (payload.get("workflow") or {}).get("result_sources")
    assert "never Resume" in text
    assert "B1000" in text and "no B1000" in text
