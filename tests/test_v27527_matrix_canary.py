from __future__ import annotations

import copy
import importlib.util
import json
import subprocess
import sys
import zipfile
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

from onnx_splitpoint_tool.benchmark.evaluation_profiles import (
    profile_model_entries,
    validate_evaluation_profile_payload,
)


ROOT = Path(__file__).resolve().parents[1]
PROFILE_ROOT = ROOT / "onnx_splitpoint_tool/resources/evaluation_profiles"
MATRIX_PROFILE = PROFILE_ROOT / "thesis_final_evaluated_matrix_v1.yaml"
GENERALIZATION_PROFILE = PROFILE_ROOT / "thesis_final_campaign_v1.yaml"
GENERATOR_PATH = ROOT / "scripts/create_v27527_yolov7_final_canary_profile.py"
LAUNCHER_PATH = ROOT / "scripts/run_v27527_yolov7_final_canary.sh"

MATRIX_MODELS = ["resnet50", "yolo26s", "yolov7_paper"]
PRODUCERS = ["hailo8", "hailo10h", "deepx"]
FULL_BACKENDS = ["tensorrt", "hailo8", "hailo10h", "deepx"]
FULL_BY_PRODUCER = {
    "hailo8": ["tensorrt", "hailo8"],
    "hailo10h": ["tensorrt", "hailo10h"],
    "deepx": ["tensorrt", "deepx"],
}


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    assert isinstance(payload, dict)
    return payload


def _load_generator() -> ModuleType:
    spec = importlib.util.spec_from_file_location(
        "v27527_yolov7_final_canary_generator", GENERATOR_PATH
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _launcher_audit_source() -> str:
    marker = (
        '"$PY" -B - "$RUN_DIR" "$DEBUG_PACK" "$LOG" '
        '"${ONNX_SPLITPOINT_REQUIRE_WARM_HIT:-0}" <<\'PY\'\n'
    )
    source = LAUNCHER_PATH.read_text(encoding="utf-8")
    assert marker in source
    return source.split(marker, 1)[1].split("\nPY\n", 1)[0]


def _run_launcher_audit_fixture(
    root: Path,
    *,
    generic_hits: dict[str, bool],
    quality_setups: set[str],
    persistent_generic_cache: bool = True,
) -> tuple[subprocess.CompletedProcess[str], dict[str, Any]]:
    run = root / "run"
    pack = root / "pack.zip"
    log = root / "run.log"
    for relative, payload in (
        (
            "campaign/campaign_readiness.json",
            {"claim_scope": "evaluated_matrix", "final_ready": True},
        ),
        (
            "reports/run_status_summary.json",
            {
                "technical_status": "ok",
                "runtime_complete": True,
                "blocking_reason_count": 0,
            },
        ),
        (
            "reports/native_evidence_status.json",
            {"technical_status": "complete", "scientific_status": "ready"},
        ),
        (
            "quality_management/central_quality_summary.json",
            {"request_count": 1, "failed_count": 0, "quality_decision": "pass"},
        ),
    ):
        path = run / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload), encoding="utf-8")

    managed_root = (
        "/home/nx/splitpoint_runs/_onnx_splitpoint_cache/"
        "tensorrt_managed_v27516/yolov7_paper-stable/b044"
    )
    generic_root = managed_root if persistent_generic_cache else "/tmp/ephemeral-trt"
    sessions = {
        name: {
            "runtime": "native_tensorrt",
            "cache_hit": hit,
            "cache_root": generic_root,
            "engine": f"{generic_root}/native/{name.replace(':', '_')}.engine",
        }
        for name, hit in generic_hits.items()
    }
    generic_result = (
        run
        / "models/yolov7_paper/benchmark_results/benchmark_results_ort_tensorrt_auto.json"
    )
    generic_result.parent.mkdir(parents=True, exist_ok=True)
    generic_result.write_text(
        json.dumps([{"native_tensorrt": {"sessions": sessions}}]),
        encoding="utf-8",
    )

    expected_setups = {
        "orin_nx_deepx_m1_01",
        "orin_nx_hailo10_01",
        "orin_nx_hailo8_01",
    }
    assert quality_setups <= expected_setups
    for setup in sorted(quality_setups):
        report = (
            run
            / "models/yolov7_paper/benchmark_results/remote_diagnostics"
            / setup
            / "case_reports/results/b044/results_split/validation_report.json"
        )
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text(
            json.dumps({
                "native_tensorrt": {
                    "sessions": {
                        "part2:tensorrt": {
                            "runtime": "native_tensorrt",
                            "cache_hit": True,
                            "build_disabled": True,
                            "engine": (
                                f"{managed_root}/native_split_quality/{setup}/"
                                "part2.engine"
                            ),
                        },
                    },
                },
            }),
            encoding="utf-8",
        )

    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("canary completed without TensorRT builds\n", encoding="utf-8")
    with zipfile.ZipFile(pack, "w") as archive:
        archive.writestr("debug_pack_manifest.json", "{}\n")
    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-c",
            _launcher_audit_source(),
            str(run),
            str(pack),
            str(log),
            "1",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    observation = json.loads(
        (run / "reports/v27527_warm_cache_observation.json").read_text(
            encoding="utf-8"
        )
    )
    return completed, observation


def _portable_source() -> dict[str, Any]:
    """Return a sealed-reference fixture with no machine-specific paths."""

    source = copy.deepcopy(_load_yaml(MATRIX_PROFILE))
    campaign = source["campaign"]
    campaign["dataset_registry"] = "sealed/dataset_registry.json"
    campaign["dataset_manifests"] = {
        "classification": {
            "calibration": "sealed/imagenet_calibration.json",
            "validation": "sealed/imagenet_validation.json",
        },
        "detection": {
            "calibration": "sealed/coco_calibration.json",
            "validation": "sealed/coco_validation.json",
        },
    }
    campaign["pipeline_contract_manifest"] = "sealed/pipeline_contract.json"
    campaign["energy_calibration_manifest"] = "sealed/fs_energy_calibration.json"

    for index, model in enumerate(source["model_suite"]["primary"], start=1):
        model["path"] = f"sealed/models/{model['id']}.onnx"
        model["model_sha256"] = f"sha256:{index:064x}"

    official = source["official_coco_evaluation"]
    official["annotations"] = "sealed/instances_val2017.json"
    official["remote_annotations"] = "sealed/remote/instances_val2017.json"
    return validate_evaluation_profile_payload(source, source="portable matrix fixture")


def _assert_matrix_scope_without_generalization(profile: dict[str, Any]) -> None:
    campaign = profile["campaign"]
    assert campaign["claim_scope"] == "evaluated_matrix"
    assert campaign["mode"] == "final"
    assert campaign["enforcement"] == "strict"
    assert campaign["frozen_before_final_campaign"] is True
    for field in (
        "require_protocol_freeze",
        "require_fitted_stage_time",
        "require_native_handover_model",
        "require_campaign_freeze",
        "require_prediction_freeze_approval",
        "prediction_freeze_enabled",
        "require_cryptographic_prediction_signature",
    ):
        assert campaign.get(field) is False, field
    assert not campaign.get("holdout_registry")
    assert not campaign.get("ranking_model_bundle")
    assert profile["ranking_validation"]["enabled"] is False
    assert profile["ranking_validation"].get("require_frozen_predictions", False) is False
    assert (
        profile["ranking_validation"].get(
            "require_complete_candidate_universe", False
        )
        is False
    )


def _assert_full_and_repetition_contract(profile: dict[str, Any]) -> None:
    native = profile["native_producers"]
    assert native["enabled"] is True
    assert native["backends"] == PRODUCERS
    assert native["consumer"] == "tensorrt"
    assert native["repetitions"] == 5
    full = native["full_baselines"]
    assert full["enabled"] is True
    assert full["required"] is True
    assert full["same_runtime_contract"] is True
    assert full["backends"] == FULL_BACKENDS
    assert full["backends_by_producer"] == FULL_BY_PRODUCER

    measurement_full = profile["measurement_campaign"]["native_full_baselines"]
    assert measurement_full == {
        "required": True,
        "same_runtime_contract": True,
        "backends": FULL_BACKENDS,
    }
    assert profile["energy"]["repeats"] == 5
    assert profile["measurement_campaign"]["system_power"]["repeats"] == 5


def _assert_serial_streaming_energy_contract(profile: dict[str, Any]) -> None:
    native_energy = profile["native_producers"]["energy"]
    assert profile["energy"]["phases"] == ["streaming"]
    assert native_energy["enabled"] is True
    assert native_energy["mode"] == "measure"
    assert native_energy["phases"] == ["streaming"]
    assert profile["workflow"]["powercalc_workers"] == 1


@pytest.mark.parametrize(
    "path,scope",
    [
        (MATRIX_PROFILE, "evaluated_matrix"),
        (GENERALIZATION_PROFILE, "ranking_generalization"),
    ],
)
def test_both_thesis_final_templates_are_schema_valid(
    path: Path, scope: str
) -> None:
    profile = _load_yaml(path)
    validated = validate_evaluation_profile_payload(profile, source=str(path))
    assert validated["campaign"]["claim_scope"] == scope
    assert validated["campaign"]["mode"] == "final"
    assert validated["campaign"]["enforcement"] == "strict"


def test_evaluated_matrix_template_has_exact_bounded_final_contract() -> None:
    profile = validate_evaluation_profile_payload(
        _load_yaml(MATRIX_PROFILE), source=str(MATRIX_PROFILE)
    )
    active = profile_model_entries(profile, include_reserve=True)
    assert [row["id"] for row in active] == MATRIX_MODELS
    assert len(active) == 3
    assert all(row.get("evaluation_role") == "development" for row in active)
    assert all(row.get("validation_tier") == "final" for row in active)

    _assert_matrix_scope_without_generalization(profile)
    selection = profile["selection_policy"]
    assert selection["max_accepted_cases_per_model"] == 1
    assert selection["preferred_shortlist"] == 5

    run_profiles = {row["id"]: row for row in profile["run_profiles"]}
    assert run_profiles["ort_tensorrt"]["full"] == "tensorrt"
    assert run_profiles["hailo8"]["full"] == "hailo8"
    assert run_profiles["hailo10"]["full"] == "hailo10"
    assert run_profiles["deepx_m1_full"]["full"] == "deepx_m1"
    _assert_full_and_repetition_contract(profile)


def test_evaluated_matrix_template_is_streaming_only() -> None:
    profile = _load_yaml(MATRIX_PROFILE)
    assert profile["energy"]["phases"] == ["streaming"]
    assert profile["native_producers"]["energy"]["phases"] == ["streaming"]


def test_evaluated_matrix_template_disables_energy_parallelism() -> None:
    profile = _load_yaml(MATRIX_PROFILE)
    assert profile["workflow"]["powercalc_workers"] == 1


def test_canary_generator_filters_only_yolov7_preserves_sealed_refs_and_final_gates() -> None:
    generator = _load_generator()
    source = _portable_source()
    canary = generator.build_canary(
        source, profile_id="portable_yolov7_final_canary"
    )
    validate_evaluation_profile_payload(canary, source="generated canary")

    active = profile_model_entries(canary, include_reserve=True)
    assert [row["id"] for row in active] == ["yolov7_paper"]
    source_yolov7 = next(
        row
        for row in source["model_suite"]["primary"]
        if row["id"] == "yolov7_paper"
    )
    assert active[0]["path"] == source_yolov7["path"]
    assert active[0]["model_sha256"] == source_yolov7["model_sha256"]
    assert active[0]["family_id"] == source_yolov7["family_id"]
    assert active[0]["evaluation_role"] == "development"
    assert active[0]["validation_tier"] == "final"

    for field in (
        "dataset_registry",
        "dataset_manifests",
        "pipeline_contract_manifest",
        "energy_calibration_manifest",
    ):
        assert canary["campaign"][field] == source["campaign"][field], field
    assert canary["official_coco_evaluation"]["annotations"] == (
        source["official_coco_evaluation"]["annotations"]
    )
    assert canary["official_coco_evaluation"]["remote_annotations"] == (
        source["official_coco_evaluation"]["remote_annotations"]
    )

    _assert_matrix_scope_without_generalization(canary)
    assert canary["quality_gate"]["frozen_before_final_campaign"] is True
    assert canary["quality_gate"]["dataset_tier"] == "final"
    assert canary["validation"]["mode"] == "strict"
    assert canary["validation"]["require_explicit"] is True
    assert canary["validation"]["require_task_metrics"] is True
    assert canary["selection_policy"]["max_accepted_cases_per_model"] == 1
    assert canary["selection_policy"]["preferred_shortlist"] == 5
    assert canary["energy"]["enabled"] is False
    assert canary["energy"]["generic_enabled"] is False
    assert canary["energy"]["requested_native_energy"] is True
    assert canary["energy"]["measurement_path"] == "native_only"
    _assert_full_and_repetition_contract(canary)
    _assert_serial_streaming_energy_contract(canary)


def test_canary_generator_rejects_ambiguous_active_yolov7() -> None:
    generator = _load_generator()
    source = _portable_source()
    duplicate = copy.deepcopy(source["model_suite"]["primary"][-1])
    duplicate["id"] = "yolov7_paper"
    duplicate["path"] = "sealed/models/yolov7_duplicate.onnx"
    source["model_suite"]["reserve"].append(duplicate)

    with pytest.raises(
        ValueError, match="expected exactly one active model with id 'yolov7_paper'"
    ):
        generator.build_canary(source, profile_id="ambiguous_yolov7_canary")


def test_canary_cli_blocks_on_missing_core_artifacts_and_writes_readiness(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    generator = _load_generator()
    source_path = tmp_path / "profile_source.yaml"
    destination = tmp_path / "yolov7_canary.yaml"
    preflight = tmp_path / "preflight"
    source_path.write_text(
        yaml.safe_dump(_portable_source(), sort_keys=False), encoding="utf-8"
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(GENERATOR_PATH),
            "--source-profile",
            str(source_path),
            "--out",
            str(destination),
            "--preflight-dir",
            str(preflight),
            "--profile-id",
            "missing_core_yolov7_canary",
        ],
    )

    assert generator.main() == 2
    generated = validate_evaluation_profile_payload(
        _load_yaml(destination), source=str(destination)
    )
    assert [
        row["id"] for row in profile_model_entries(generated, include_reserve=True)
    ] == ["yolov7_paper"]

    readiness = json.loads(
        (preflight / "campaign_readiness.json").read_text(encoding="utf-8")
    )
    assert readiness["claim_scope"] == "evaluated_matrix"
    assert readiness["status"] == "blocked"
    assert readiness["final_ready"] is False
    assert readiness["ready"] is False
    assert readiness["required_failure_count"] > 0
    checks = {row["id"]: row for row in readiness["checks"]}
    for check_id in (
        "dataset_classification_calibration",
        "dataset_classification_validation",
        "dataset_detection_calibration",
        "dataset_detection_validation",
        "pipeline_contract_manifest",
        "model_identity_yolov7_paper",
        "energy_calibration_manifest",
    ):
        assert checks[check_id]["status"] == "fail", check_id
    for check_id in (
        "holdout_models_present",
        "holdout_registry",
        "ranking_validation_enabled",
        "ranking_model_bundle_integrity",
        "stage_time_model",
        "native_handover_model",
    ):
        assert checks[check_id]["status"] == "not_applicable", check_id


def test_launcher_warm_gate_requires_positive_persistent_hits_for_every_session(
    tmp_path: Path,
) -> None:
    session_names = (
        "full:tensorrt",
        "part1:tensorrt",
        "part2:tensorrt",
    )
    setup_ids = {
        "orin_nx_deepx_m1_01",
        "orin_nx_hailo10_01",
        "orin_nx_hailo8_01",
    }
    all_hits = {name: True for name in session_names}

    passed, observation = _run_launcher_audit_fixture(
        tmp_path / "all_hits",
        generic_hits=all_hits,
        quality_setups=setup_ids,
    )
    assert passed.returncode == 0, passed.stderr
    assert "PHYSICAL_TRT_BUILDS=0" in passed.stdout
    assert "WARM_CACHE_CANDIDATE=PASS" in passed.stdout
    assert observation["zero_physical_trt_builds"] is True
    assert observation["generic_cross_run_hit_evidence_complete"] is True
    assert observation["quality_cross_run_hit_evidence_complete"] is True
    assert observation["warm_cache_candidate"] is True

    for missing_hit in session_names:
        hits = dict(all_hits)
        hits[missing_hit] = False
        failed, failed_observation = _run_launcher_audit_fixture(
            tmp_path / f"miss_{missing_hit.split(':', 1)[0]}",
            generic_hits=hits,
            quality_setups=setup_ids,
        )
        assert failed.returncode != 0
        assert "PHYSICAL_TRT_BUILDS=0" in failed.stdout
        assert failed_observation["zero_physical_trt_builds"] is True
        assert failed_observation["generic_cross_run_hit_evidence_complete"] is False
        assert failed_observation["warm_cache_candidate"] is False

    ephemeral, ephemeral_observation = _run_launcher_audit_fixture(
        tmp_path / "ephemeral",
        generic_hits=all_hits,
        quality_setups=setup_ids,
        persistent_generic_cache=False,
    )
    assert ephemeral.returncode != 0
    assert ephemeral_observation["generic_cross_run_hit_evidence_complete"] is False

    incomplete_quality, quality_observation = _run_launcher_audit_fixture(
        tmp_path / "missing_quality_setup",
        generic_hits=all_hits,
        quality_setups=setup_ids - {"orin_nx_hailo10_01"},
    )
    assert incomplete_quality.returncode != 0
    assert quality_observation["quality_cross_run_hit_evidence_complete"] is False
    assert quality_observation["warm_cache_candidate"] is False
