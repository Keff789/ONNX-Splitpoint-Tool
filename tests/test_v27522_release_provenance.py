from __future__ import annotations

from pathlib import Path
import re

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot


ROOT = Path(__file__).resolve().parents[1]
CURRENT_VERSION = "2.75.47"
CURRENT_BUILD = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"


def test_current_release_identity_retains_v27522_features() -> None:
    assert tool.__version__ == CURRENT_VERSION
    assert tool.__release__ == CURRENT_VERSION
    assert tool.__development_lineage__ == f"v{CURRENT_VERSION}"
    assert tool.__build_id__ == CURRENT_BUILD
    assert WORKFLOW_VERSION == CURRENT_BUILD
    assert {
        "pre_mutation_remote_failure_collection_suppression",
        "remote_connectivity_primary_error_preservation",
        "post_lease_cleanup_quarantine_fail_closed",
        "packaged_resnet_v27539_deepx_full_quality_canary",
        "deepx_full_quality_only_dxnn_dispatch",
        "deepx_full_quality_cpu_fallback_forbidden",
        "full_only_quality_performance_matrix_not_applicable",
        "deepx_full_canary_hailo_gate_not_applicable",
        "packaged_resnet_v27538_deepx_full_quality_canary",
        "packaged_resnet_v27537_deepx_full_quality_canary",
        "backend_artifact_diagnostic_debug_pack",
        "native_singleton_shape_validator_equivalence",
        "stable_trt_engine_cache_namespace",
        "verified_legacy_trt_engine_cache_migration",
        "jetson_cuda_driver_trt_abi_attestation",
        "selected_gpu_trt_engine_cache_identity",
        "warm_trt_retention_metadata_fast_path",
        "quality_first_no_build_receipt_preserving_reuse",
        "disabled_energy_not_applicable_evidence",
        "packaged_resnet_v27536_acceptance_profile",
        "hailo10_singleton_boundary_layout_resolution",
        "native_split_quality_case_failure_isolation",
        "frozen_audit_execution_union_generation",
        "frozen_audit_union_identity_preflight",
        "audit_scoped_standard_development_ranking",
        "model_validation_summary_debug_pack",
        "score_independent_audit_execution_plan",
        "native_completed_task_ranking_strata",
        "development_screening_ranking_analysis",
        "truthful_native_ranking_audit_status",
        "known_alias_debug_pack_deduplication",
        "compact_canonical_debug_pack",
        "full_workflow_log_debug_pack",
        "native_ranking_observation_ingestion",
        "simplified_model_usage_profile_editor",
        "score_independent_profile_schema",
        "canonical_native_tensorrt_performance_alias",
        "diagnostic_energy_screening_comparability_demotion",
        "standard_identity_full_only_canary_cache_probe",
        "receipt_attested_hailo_full_preupload_gate",
        "failed_backend_artifact_upload_block",
        "custom_evaluation_profile_update_preservation",
        "run_mode_full_only_canary_preservation",
        "quality_canary_variant_recipe_validation",
        "yolov7_full_only_canary_launcher",
        "detection_ap75_guardrail",
        "configured_guardrail_fail_closed",
        "offline_quality_prediction_replay",
        "setup_local_quality_report_identity",
        "technical_quality_status_separation",
        "explicit_full_only_quality_canary",
        "debug_pack_quality_provenance_receipts",
    } <= set(tool.__build_features__)


def test_package_and_lock_root_match_current_release_identity() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert re.search(
        rf'^version\s*=\s*"{re.escape(CURRENT_VERSION)}"$',
        pyproject,
        re.MULTILINE,
    )
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    package = re.search(
        r'\[\[package\]\]\nname = "onnx-splitpoint-tool"\nversion = "([^"]+)"',
        lock,
    )
    assert package is not None
    assert package.group(1) == CURRENT_VERSION


def test_current_guides_and_historical_v27522_canary_are_preserved() -> None:
    guide = (
        ROOT / f"TESTANLEITUNG_{CURRENT_VERSION}.md"
    ).read_text(encoding="utf-8")
    report = (
        ROOT
        / f"VERSION_{CURRENT_VERSION}_BUILD_AND_TEST_REPORT.md"
    ).read_text(encoding="utf-8")
    wrapper = (
        ROOT / "scripts/run_v27524_yolov7_quality_canary.sh"
    ).read_text(encoding="utf-8")
    replay = (
        ROOT / "scripts/run_v27524_yolov7_pack_replay.sh"
    ).read_text(encoding="utf-8")
    for text in (guide, report):
        assert CURRENT_VERSION in text
        assert CURRENT_BUILD in text
        assert "Final Quality" in text
        assert "5.000" in text
        assert "Standard" in text
    current_small = ROOT / "scripts/run_v27540_small_acceptance.sh"
    assert current_small.is_file()
    assert "deepx_preprocessing_ab_paired_cohort_lock" in current_small.read_text(
        encoding="utf-8"
    )
    assert "ONNX_SPLITPOINT_CANARY_MIN_FREE_GIB:-15" in wrapper
    assert "--require-fresh-run" in wrapper
    assert "create_evaluation_debug_pack.py" in wrapper
    assert "rm -rf -- \"$RUN_DIR\"" not in wrapper
    assert "--workers 4" in replay
    assert "--full-only" in replay
    assert "central_quality_replay_v27522.json" in replay


def test_updater_preserves_runtime_environment_and_caches() -> None:
    updater = (ROOT / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    assert "--exclude='/.venv/'" in updater
    assert "--exclude='/EvaluationRuns/'" in updater
    assert "--exclude='/RemoteBenchmarkRuns/'" in updater
    assert "--exclude='/BenchmarkSets/'" in updater
    assert "PRESERVED_PROFILE_DIR" in updater
    assert "restore_preserved_profiles" in updater
    assert "--backup-dir" in updater
    assert "--verify-archive" in updater


def test_claim_critical_identity_covers_new_replay_and_canary_modules() -> None:
    build = package_build_snapshot()
    critical = set(build["critical_module_sha256"])

    assert build["critical_module_count"] >= 165
    assert build["critical_module_set_complete"] is True
    assert {
        "quality_replay.py",
        "workflow/full_only_quality_canary.py",
        "workflow/setup_local_trt_dispatch.py",
    } <= critical
