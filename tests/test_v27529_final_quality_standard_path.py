from __future__ import annotations

import copy
import json
from pathlib import Path

import onnx_splitpoint_tool.campaign as campaign_module
import pytest
import yaml
from onnx_splitpoint_tool.campaign import build_campaign_readiness
from onnx_splitpoint_tool.execution_plan import build_effective_execution_plan
from onnx_splitpoint_tool.run_modes import (
    RUN_MODE_SCHEMA,
    RUN_MODE_SCHEMA_VERSION,
    _legacy_v11_final_mode,
    apply_run_mode,
    default_run_modes_config,
    load_run_modes_config,
    official_coco_policy_for_mode,
    run_modes_revision,
    save_run_modes_config,
    validate_run_modes_config,
)
from onnx_splitpoint_tool.workflow.runner import (
    _effective_evidence_run_mode_v27529,
    _native_energy_final_contract_requested_v61d,
)


def _source(mode_id: str, *, energy: bool = True) -> dict:
    config = default_run_modes_config()
    return {
        "name": "final_quality_standard_path_test",
        "model_suite": {
            "primary": [
                {
                    "id": "resnet50",
                    "task": "classification",
                    "evaluation_role": "development",
                }
            ]
        },
        "selection_policy": {
            "max_accepted_cases_per_model": 2,
            "preferred_shortlist": 5,
            "selection_strategy": "stratified_windows",
        },
        "run_profiles": [
            {
                "id": "ort_tensorrt",
                "type": "same_backend_reference",
                "full": "tensorrt",
                "stage1": "tensorrt",
                "stage2": "tensorrt",
                "required": True,
            }
        ],
        "execution_preset": {
            "id": mode_id,
            "follow_tool_config": False,
            "snapshot": config["modes"][mode_id],
            "overrides": {
                "native_enabled": True,
                "energy_enabled": energy,
            },
        },
    }


def _resolved(mode_id: str, *, energy: bool = True) -> dict:
    config = default_run_modes_config()
    profile, _audit = apply_run_mode(
        _source(mode_id, energy=energy),
        config=config,
    )
    return profile


def test_final_quality_is_mechanically_standard_plus_quality_only() -> None:
    config = default_run_modes_config()
    assert RUN_MODE_SCHEMA_VERSION == 13
    standard = config["modes"]["standard"]
    final = config["modes"]["final"]

    expected = copy.deepcopy(standard)
    expected.update({
        "label": "Final Quality (Standard+)",
        "description": final["description"],
        "recommended_for": final["recommended_for"],
    })
    expected["data"]["validation_items"] = {
        "classification": 5000,
        "detection": 5000,
    }
    expected["quality"].update({
        "profile_id": "task_quality_final_5000",
        "dataset_tier": "final",
        "bootstrap_repetitions": 5000,
    })

    assert final == expected
    assert final["campaign"]["mode"] == "development"
    assert final["campaign"]["enforcement"] == "warn"
    assert final["reproducibility"]["level"] == "relaxed"
    assert final["reproducibility"]["dataset_sample_size"] == 24
    assert final["build"]["hailo"]["preset"] == "balanced"
    assert final["build"]["hailo"]["optimization_level"] == 1
    assert final["runtime"]["benchmark"] == {
        "provider": "auto", "warmup": 3, "runs": 5, "timeout_s": 0,
    }
    assert final["runtime"]["native"]["frames"] == 1000
    assert final["runtime"]["native"]["warmup"] == 100
    assert final["runtime"]["native"]["repetitions"] == 3
    assert final["energy"] == standard["energy"]


@pytest.mark.parametrize("energy", [False, True])
def test_materialized_final_quality_uses_standard_execution_and_energy(
    energy: bool,
) -> None:
    standard = _resolved("standard", energy=energy)
    final = _resolved("final", energy=energy)

    unchanged_blocks = (
        "integrity_policy",
        "campaign",
        "hailo_build",
        "deepx_build",
        "artifact_store",
        "build_scheduler",
        "benchmark_execution",
        "remote_cache",
        "native_producers",
        "ranking_validation",
        "reporting",
        "energy",
        "official_coco_evaluation",
    )
    for key in unchanged_blocks:
        assert final[key] == standard[key], key
    assert final["model_preparation"]["mode"] == standard["model_preparation"]["mode"]
    assert final["hardware_smoke"]["mode"] == standard["hardware_smoke"]["mode"]
    assert final["hardware_smoke"]["require_remote_for_hailo"] is False

    assert standard["validation_execution"]["max_items"] == {
        "classification": 500,
        "detection": 500,
    }
    assert final["validation_execution"]["max_items"] == {
        "classification": 5000,
        "detection": 5000,
    }
    assert standard["quality_gate"]["statistics"]["bootstrap_repetitions"] == 500
    assert final["quality_gate"]["statistics"]["bootstrap_repetitions"] == 5000
    assert final["quality_gate"]["dataset_tier"] == "final"
    assert final["campaign"]["mode"] == "development"
    assert final["integrity_policy"]["dataset_sample_size"] == 24

    standard_plan = build_effective_execution_plan(standard)
    final_plan = build_effective_execution_plan(final)
    execution_fields = (
        "generic_rows_total",
        "remote_run_invocations_total",
        "cold_suite_uploads_total",
        "native_enabled",
        "native_energy_enabled",
        "benchmark_warmup",
        "benchmark_runs",
        "native_frames",
        "native_warmup",
        "native_performance_repetitions",
        "native_queue_depth",
        "native_inflight",
        "hailo_preset",
        "hailo_optimization_level",
    )
    for key in execution_fields:
        assert final_plan[key] == standard_plan[key], key

    def _diff_paths(left, right, path=""):
        if isinstance(left, dict) and isinstance(right, dict):
            differences = set()
            for key in set(left) | set(right):
                child = f"{path}/{key}"
                if key not in left or key not in right:
                    differences.add(child)
                else:
                    differences.update(
                        _diff_paths(left[key], right[key], child)
                    )
            return differences
        return {path} if left != right else set()

    assert _diff_paths(standard_plan, final_plan) == {
        "/bootstrap_repetitions",
        "/estimated_validation_files_total",
        "/native_execution_contract/contract_sha256",
        "/native_execution_contract/run_mode",
        "/native_execution_contract_sha256",
        "/run_mode",
        "/run_mode_label",
        "/validation_items/classification",
        "/validation_items/detection",
    }
    assert official_coco_policy_for_mode("final") == (
        official_coco_policy_for_mode("standard")
    )
    assert official_coco_policy_for_mode("final")["archive_eval_tensors"] is False


def test_final_quality_does_not_activate_strict_final_energy_or_evidence() -> None:
    final = _resolved("final", energy=True)
    energy_cfg = final["native_producers"]["energy"]
    assert _native_energy_final_contract_requested_v61d(final, energy_cfg) is False
    assert _effective_evidence_run_mode_v27529(final) == "standard"

    strict_campaign = copy.deepcopy(final)
    strict_campaign["campaign"]["mode"] = "final"
    assert _native_energy_final_contract_requested_v61d(
        strict_campaign, energy_cfg,
    ) is True
    assert _effective_evidence_run_mode_v27529(strict_campaign) == "final"


def test_schema_v11_registry_migrates_final_and_preserves_custom_leaf(
    tmp_path: Path,
) -> None:
    defaults = default_run_modes_config()
    legacy = {
        "schema": RUN_MODE_SCHEMA,
        "schema_version": 11,
        "default_mode": "standard",
        "modes": {
            "smoke": copy.deepcopy(defaults["modes"]["smoke"]),
            "standard": copy.deepcopy(defaults["modes"]["standard"]),
            "final": _legacy_v11_final_mode(),
        },
    }
    migrated = validate_run_modes_config(legacy)
    assert migrated["schema_version"] == 13
    assert migrated["modes"]["final"] == defaults["modes"]["final"]

    customised = copy.deepcopy(legacy)
    customised["modes"]["final"]["runtime"]["benchmark"]["runs"] = 7
    migrated_custom = validate_run_modes_config(customised)
    assert migrated_custom["modes"]["final"]["runtime"]["benchmark"]["runs"] == 7
    assert migrated_custom["modes"]["final"]["campaign"]["mode"] == "development"
    assert migrated_custom["modes"]["final"]["quality"]["bootstrap_repetitions"] == 5000

    registry = tmp_path / "run_modes.yaml"
    registry.write_text(
        yaml.safe_dump(legacy, sort_keys=False), encoding="utf-8",
    )
    before = registry.read_bytes()
    loaded = load_run_modes_config(registry)
    assert loaded["modes"]["final"] == defaults["modes"]["final"]
    assert registry.read_bytes() == before
    save_run_modes_config(registry, loaded, expected_revision=run_modes_revision(loaded))
    persisted = yaml.safe_load(registry.read_text(encoding="utf-8"))
    assert persisted["schema_version"] == 13
    assert persisted["modes"]["final"] == defaults["modes"]["final"]


def test_schema_v10_untouched_final_migrates_directly_to_standard_plus() -> None:
    defaults = default_run_modes_config()
    historical_final = _legacy_v11_final_mode()
    historical_final["description"] = (
        "Maximum-effort frozen campaign with 5,000 validation items per task "
        "by default, strict provenance and thesis reporting."
    )
    historical_final["recommended_for"] = (
        "Thesis tables, hold-out claims, native/full baselines and calibrated "
        "system-energy results."
    )
    historical_campaign = historical_final["campaign"]
    historical_campaign.pop("claim_scope", None)
    historical_campaign.pop("require_protocol_freeze", None)
    historical_campaign.update({
        "require_fitted_stage_time": True,
        "require_native_handover_model": True,
        "require_campaign_freeze": True,
        "require_prediction_freeze_approval": True,
    })
    historical_final["ranking"]["enabled"] = True
    historical_final["holdout"].update({
        "prediction_freeze_enabled": True,
        "require_complete_candidate_universe": True,
        "require_frozen_predictions": True,
        "require_unseen_attestation": True,
    })
    historical_final["energy"]["phases"] = ["latency", "streaming"]
    legacy = {
        "schema": RUN_MODE_SCHEMA,
        "schema_version": 10,
        "default_mode": "standard",
        "modes": {
            "smoke": copy.deepcopy(defaults["modes"]["smoke"]),
            "standard": copy.deepcopy(defaults["modes"]["standard"]),
            "final": historical_final,
        },
    }

    migrated = validate_run_modes_config(legacy)
    assert migrated["modes"]["final"] == defaults["modes"]["final"]


def test_final_quality_campaign_readiness_uses_same_sampled_check_as_standard(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manifest_paths: dict[str, dict[str, str]] = {
        "classification": {},
        "detection": {},
    }
    for task in ("classification", "detection"):
        for role in ("calibration", "validation"):
            path = tmp_path / f"{task}_{role}.json"
            path.write_text(json.dumps({
                "schema": "onnx-splitpoint/dataset-manifest",
                "schema_version": 1,
                "task": task,
                "role": role,
                "hash_mode": "content",
                "items_identity_sha256": "a" * 64,
                "item_count": 1,
                "items": [],
            }), encoding="utf-8")
            manifest_paths[task][role] = str(path)

    calls: list[tuple[str, int]] = []

    def _verify(_payload, *, verify_files, sample_size, verification_mode):
        assert verify_files is True
        calls.append((verification_mode, sample_size))
        return {"ok": True}

    monkeypatch.setattr(campaign_module, "verify_dataset_manifest", _verify)
    monkeypatch.setattr(
        campaign_module,
        "validate_calibration_validation_separation",
        lambda _manifests: {"ok": True},
    )

    for mode_id in ("standard", "final"):
        profile = _resolved(mode_id, energy=False)
        profile["campaign"]["dataset_manifests"] = copy.deepcopy(manifest_paths)
        calls.clear()
        report = build_campaign_readiness(
            profile,
            profile_path=tmp_path / f"{mode_id}.yaml",
        )
        assert report["mode"] == "development"
        assert calls == [("sampled", 24)] * 4


def test_packaged_registry_and_normal_workflow_have_no_hidden_campaign_bootstrap() -> None:
    root = Path(__file__).resolve().parents[1]
    packaged = yaml.safe_load(
        (root / "onnx_splitpoint_tool/resources/run_modes/default_run_modes.yaml")
        .read_text(encoding="utf-8")
    )
    assert packaged == default_run_modes_config()

    forbidden = (
        "prepare_evaluated_matrix_campaign",
        "create_v27527_yolov7_final_canary_profile",
    )
    normal_sources = (
        root / "onnx_splitpoint_tool/gui/profile_editor.py",
        root / "onnx_splitpoint_tool/benchmark/evaluation_profiles.py",
        root / "onnx_splitpoint_tool/workflow/runner.py",
        root / "onnx_splitpoint_tool/workflow/run_evaluation.py",
    )
    for source in normal_sources:
        text = source.read_text(encoding="utf-8")
        assert not any(marker in text for marker in forbidden), source
