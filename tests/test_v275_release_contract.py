from __future__ import annotations

import json
import os
from pathlib import Path
import re
import subprocess
import sys

import onnx_splitpoint_tool as tool
from onnx_splitpoint_tool.preprocessing_contract import (
    canonical_image_preprocessing_contract,
    preprocessing_contract_sha256,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot


ROOT = Path(__file__).resolve().parents[1]
VERSION = "2.75.47"
BUILD = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
HISTORICAL_V27546_VERSION = "2.75.46"
HISTORICAL_V27546_BUILD = (
    "v2.75.46-native-full-onnx-attestation-standard-quality-projection"
)
HISTORICAL_V27528_VERSION = "2.75.28"
HISTORICAL_V27528_BUILD = (
    "v2.75.28-final-campaign-bootstrap-energy-provenance-repair"
)
HISTORICAL_FULL_ONLY_VERSION = "2.75.21"
HISTORICAL_FULL_ONLY_BUILD = (
    "v2.75.21-setup-local-tensorrt-quality-dispatch-repair"
)


def test_v275_release_identity_and_features_are_sealed() -> None:
    assert tool.__version__ == VERSION
    assert tool.__release__ == VERSION
    assert tool.__development_lineage__ == f"v{VERSION}"
    assert tool.__build_id__ == BUILD
    assert WORKFLOW_VERSION == BUILD
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
        "frozen_audit_rejection_continuation",
        "profile_selection_budget_roundtrip",
        "packaged_resnet_small_acceptance_profile",
        "hailo10_singleton_boundary_layout_resolution",
        "native_split_quality_case_failure_isolation",
        "frozen_audit_execution_union_generation",
        "frozen_audit_union_identity_preflight",
        "audit_scoped_standard_development_ranking",
        "model_validation_summary_debug_pack",
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
        "logical_native_split_plan_authority",
        "full_only_zero_split_native_matrix",
        "full_only_native_multi_input_admission",
        "automatic_management_cpu_reference_recipe",
        "tensorrt_full_output_contract_projection",
        "explicit_run_profile_target_authority",
        "deepx_full_sealed_input_contract_projection",
        "collected_native_full_semantic_path_rebase",
        "terminal_full_semantic_fail_closed",
        "native_runtime_policy_alias_projection",
        "gui_offline_replay_core_preservation",
        "vendor_runtime_policy_hash_diagnostic_only",
        "hailo8_yolov7_canonical_raw_head_loader_chain",
        "hailo_full_management_free_remote_closure",
        "runtime_plan_post_normalization_finalization",
        "three_node_native_contract_preflight",
        "failure_safe_full_only_wrapper_reporting",
        "setup_local_tensorrt_quality_companion_dispatch",
        "deepx_single_tensorrt_performance_owner",
        "prethread_setup_local_tensorrt_dispatch_preflight",
        "matrix_bound_native_summary_denominator",
        "setup_local_tensorrt_failure_classification",
        "full_only_split_quality_not_applicable",
        "task_bound_preprocessing_contract_v2",
        "canonical_detection_letterbox_rgb_pad114",
        "hailo_preprocessing_bound_cache_receipt",
        "runtime_attested_yolov7_raw_full_endpoint",
        "completed_v2_persisted_self_reference",
        "fail_closed_explicit_empty_selection",
        "run_mode_native_configuration_preservation",
        "disabled_energy_not_applicable_completion",
        "exact_vendor_full_quality_provenance",
        "deepx_bn6_completed_postfilter_parity",
        "fail_closed_hailo_task_resolution",
        "materialized_hailo_hef_receipt_binding",
        "hailo_bn6_completed_hotloop_parity",
        "portable_completed_artifact_rebase",
        "persisted_detection_execution_v1_reference",
        "mandatory_vendor_full_quality_binding_matrix",
        "runtime_preprocessing_identity_sha256",
        "hailo_runtime_receipt_verification",
        "strict_deepx_bn6_record_semantics",
        "strict_hailo_receipt_cache_chain",
        "exact_hailo_physical_receipt_binding",
        "receipt_sealed_hailo_raw_end_nodes",
        "stale_hailo_claim_demotion",
        "target_hw_trt_quality_contract_bridge",
        "post_build_hailo_receipt_promotion",
        "shared_pre_timing_deepx_input_sealer",
        "row_local_quality_claim_veto",
        "version_neutral_hailo_exact_v2_cache_restore",
        "version_neutral_hailo_v3_with_v2_migration",
        "zero_start_native_reporting",
        "stat_collision_content_probe_cache",
        "remote_native_full_dependency_closure",
        "row_terminal_native_checkpoint_continue",
        "collection_receipt_guarded_remote_cleanup",
        "receipt_bound_source_graph_raw_head_attestation",
        "partial_runtime_native_energy_observation",
        "canonical_native_execution_contract",
        "cache_verify_only_artifact_policy",
        "cache_verify_only_exact_plan_attestation",
        "cache_verify_only_compiler_dispatch_fence",
        "cache_verify_only_canary_profile",
        "cache_verify_managed_venv_compiler_identity",
        "cache_verify_exact_case_failfast",
        "cache_verify_artifact_scope_attestation",
        "cache_verify_full_receipt_predispatch",
        "cache_verify_structured_miss_diagnostic",
        "cache_verify_nested_suite_resolution",
        "cache_verify_native_split_cache_replay",
        "cache_verify_exact_terminal_postcondition",
        "cache_verify_no_management_ort",
        "cache_verify_validation_not_applicable",
        "frozen_variant_remote_binding",
        "synchronized_source_update_preserves_venv",
        "profile_driven_cache_canary_run_id",
        "cache_canary_complete_remote_runtime_closure",
        "attested_smoke_cache_identity",
        "cross_suite_exact_cache_resolution",
        "invalid_cache_binding_not_identity_miss",
        "canonical_native_result_verification",
        "optional_native_result_status_alias",
        "sealed_deepx_full_image_identity",
        "receipt_attested_hailo_full_contract_overlay",
        "native_full_child_failure_diagnostics",
        "exact_native_full_line_verifier",
        "quality_join_nearest_axis_diagnostics",
        "vendor_full_nine_row_acceptance",
        "portable_deepx_full_semantic_manifest",
        "sealed_deepx_quality_model_identity",
        "idempotent_hailo_full_contract_promotion",
        "canonical_quality_record_endpoint_contract",
        "owned_managed_trt_cache_retention_contract",
        "same_host_remote_storage_serialization",
        "campaign_preflight_success_exit_contract",
        "exact_yolov7_paper_final_template",
        "runtime_exact_campaign_contract_templates",
        "yolov7_paper_final_canary",
        "inherited_energy_method_runtime_binding_gate",
    } <= set(tool.__build_features__)


def test_v275_detection_and_classification_contracts_are_task_bound() -> None:
    detection = canonical_image_preprocessing_contract("detection", (640, 640))
    classification = canonical_image_preprocessing_contract(
        "classification", (224, 224)
    )
    assert detection["preprocess_mode"] == "letterbox"
    assert detection["spatial_transform"] == "centered_letterbox"
    assert detection["color_space"] == "RGB"
    assert detection["pad_value"] == 114
    assert detection["letterbox"] is True
    assert classification["preprocess_mode"] == "resize"
    assert classification["image_scale"] == "imagenet"
    assert classification["letterbox"] is False
    assert len(preprocessing_contract_sha256(detection)) == 64
    assert preprocessing_contract_sha256(detection) != preprocessing_contract_sha256(
        classification
    )


def test_v275_current_release_documents_are_present() -> None:
    guide = ROOT / "TESTANLEITUNG_2.75.47.md"
    report = ROOT / "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"
    assert guide.is_file()
    assert report.is_file()
    for path in (ROOT / "README.md", ROOT / "docs/VERSIONING.md", guide, report):
        text = path.read_text(encoding="utf-8")
        assert VERSION in text
        assert BUILD in text
    assert (ROOT / "scripts/replay_central_quality.py").is_file()
    assert (ROOT / "scripts/run_v27539_small_acceptance.sh").is_file()
    assert (
        ROOT / "profiles" / "resnet50_v27539_deepx_full_quality_canary.yaml"
    ).is_file()
    assert (ROOT / "scripts/run_v27536_small_acceptance.sh").is_file()
    assert (
        ROOT / "profiles" / "resnet50_v27536_small_acceptance.yaml"
    ).is_file()
    assert (ROOT / "scripts/run_v27528_small_acceptance.sh").is_file()
    assert (ROOT / "scripts/run_v27524_yolov7_quality_canary.sh").is_file()
    assert (ROOT / "scripts/run_v27524_yolov7_pack_replay.sh").is_file()
    assert (ROOT / "scripts/run_v27526_small_acceptance.sh").is_file()
    assert (ROOT / "scripts/run_v27524_small_acceptance.sh").is_file()
    assert (ROOT / "scripts/run_v27522_yolov7_offline_replay.sh").is_file()
    assert (ROOT / "scripts/run_v27522_small_acceptance.sh").is_file()
    assert not (ROOT / "TESTANLEITUNG_2.75.14.md").exists()
    assert not (ROOT / "VERSION_2.75.14_BUILD_AND_TEST_REPORT.md").exists()


def test_v27529_documents_describe_standard_path_and_retain_v27528_tools() -> None:
    guide = (ROOT / "TESTANLEITUNG_2.75.46.md").read_text(encoding="utf-8")
    report = (ROOT / "VERSION_2.75.46_BUILD_AND_TEST_REPORT.md").read_text(
        encoding="utf-8",
    )
    for text in (guide, report):
        assert HISTORICAL_V27546_VERSION in text
        assert HISTORICAL_V27546_BUILD in text
        assert "Final Quality" in text
        assert "5.000" in text
        assert "Standard" in text
    assert "run_v27528_*_final_canary.sh" in guide

    historical_small = (
        ROOT / "scripts/run_v27528_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    historical_generator = (
        ROOT / "scripts/create_v27528_yolov7_final_canary_profile.py"
    ).read_text(encoding="utf-8")
    assert HISTORICAL_V27528_VERSION in historical_small
    assert HISTORICAL_V27528_BUILD in historical_small
    assert "yolov7_paper" in historical_generator
    assert "final_ready" in historical_generator


def test_v275_acceptance_allows_resolved_legacy_failures_but_not_new_ones() -> None:
    script = (ROOT / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert f'EXPECTED_VERSION="{VERSION}"' in script
    assert f'EXPECTED_BUILD="{BUILD}"' in script
    assert "new_failures = sorted(observed - expected)" in script
    assert "if new_failures:" in script
    assert "if observed != expected:" not in script


def test_v275_update_and_canary_scripts_preserve_venv_and_find_python() -> None:
    updater = (ROOT / "scripts/update_source_release.sh").read_text(
        encoding="utf-8"
    )
    canary = (ROOT / "scripts/run_v27521_cache_canary.sh").read_text(
        encoding="utf-8"
    )
    assert "--delete" in updater
    assert "--backup" in updater
    assert "--exclude='/.venv/'" in updater
    assert "PRESERVED_PROFILE_DIR" in updater
    assert "restore_preserved_profiles" in updater
    assert "-iname '*.yaml'" in updater
    assert "--verify-archive" in updater
    assert "build_source_manifest.py" in updater
    assert 'PYTHON="$TOOL_DIR/.venv/bin/python"' in updater
    assert "NORMALIZED_USER_HOME" in updater
    assert "SOURCE_MANIFEST.json" in updater
    assert 'PY="$PYTHON" bash scripts/run_v27521_small_acceptance.sh' in canary
    assert '.venv/bin/python' in canary
    assert "run_v27521_small_acceptance.sh" in canary
    assert "cache_verify_resnet50_b052_hailo8.yaml" in canary
    assert "--require-run-mode smoke" in canary
    assert "--require-run-mode standard" not in canary
    small = (ROOT / "scripts/run_v27521_small_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert "tests/test_v27510_smoke_cache_compatibility.py" in small
    assert "tests/test_v27511_cache_verify_semantic_path.py" in small
    assert "tests/test_v27511_canonical_native_collector.py" in small
    assert "tests/test_v27511_semantic_cache_canary_verifier.py" in small
    assert "tests/test_v2752_small_acceptance.py" in small
    assert "tests/test_v270b_hardware_smoke_repair.py" in small
    assert "tests/test_v275_deepx_full_quality_contract.py" in small
    assert "tests/test_v275_deepx_sealed_runtime_input.py" in small
    assert "tests/test_v27520_remote_boundary_and_plan_finalization.py" in small
    current_small = (ROOT / "scripts/run_v27528_small_acceptance.sh").read_text(
        encoding="utf-8"
    )
    for name in (
        "test_v27528_campaign_preparation.py",
        "test_v27528_energy_inherited_runtime_binding.py",
        "test_v27528_template_canary.py",
        "test_v27527_claim_scope.py",
        "test_v27527_matrix_canary.py",
        "test_v60n_run_modes.py",
        "test_v275_release_contract.py",
        "test_v273_local_acceptance_harness.py",
    ):
        assert name in current_small
    replay = (
        ROOT / "scripts/run_v27522_yolov7_offline_replay.sh"
    ).read_text(encoding="utf-8")
    assert "--workers 4" in replay
    assert "--full-only" not in replay
    assert "_offline_quality_replay_v27522" in replay


def test_v27521_full_only_check_is_fresh_and_strictly_eighteen_rows() -> None:
    script = (
        ROOT / "scripts/run_v27521_full_only_check.sh"
    ).read_text(encoding="utf-8")
    assert "--require-run-mode smoke" in script
    assert "--require-fresh-run" in script
    assert "--execution-mode generate_and_run" in script
    assert "--run-id" not in script
    assert "run_manifest.json" in script
    assert "run_status_summary.json" in script
    assert '{"ok", "partial"}' in script
    assert '{"ok", "partial", "failed"}' not in script
    assert "len(candidates) != 1" in script
    assert "full-only-run-directory-snapshot" in script
    assert "preexisting_directory_name" in script
    assert "preexisting_directory_identity" in script
    assert '"fresh"' in script
    assert "--scope all" in script
    assert "--scope vendor" not in script
    assert "passed=18/18" in script
    assert "(0, 18, 18)" in script
    assert 'stage.get("split_backends") != []' in script
    assert "management_cpu_reference_status.json" in script
    assert "expected_run_ids" in script
    assert "runs != planned_runs" in script
    assert "management_cpu_reference_invariant" in script
    assert "run_plan_sha256" in script
    assert "benchmark_set.json spiegelt den Runplan nicht" in script
    assert 'row.get("backend") == "cuda_ort"' in script
    assert "set -Eeuo pipefail" in script
    assert script.index("scripts/verify_native_full_rows.py") < script.index(
        "PASS Full-only passed=18/18"
    )
    assert "flock" not in script
    assert "hardware-lock" not in script.lower()
    checker = (ROOT / "scripts/verify_native_full_rows.py").read_text(
        encoding="utf-8"
    )
    assert "EXPECTED_VENDOR_FULL_IDENTITIES" in checker
    assert "expected_{scope}_full_matrix_not_exact" in checker


def test_v27521_full_only_profile_preflight_rejects_extra_logical_axes(
    tmp_path: Path,
) -> None:
    script = (
        ROOT / "scripts/run_v27521_full_only_check.sh"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"<<'PROFILE_PY'\n(?P<code>.*?)\nPROFILE_PY",
        script,
        flags=re.DOTALL,
    )
    assert match is not None
    profile_code = match.group("code")
    run_profiles = [
        {
            "id": "ort_tensorrt", "type": "same_backend_reference",
            "full": "tensorrt", "stage1": "tensorrt",
            "stage2": "tensorrt",
        },
        {
            "id": "hailo8", "type": "same_backend_reference",
            "full": "hailo8", "stage1": "hailo8", "stage2": "hailo8",
        },
        {
            "id": "hailo10", "type": "same_backend_reference",
            "full": "hailo10", "stage1": "hailo10",
            "stage2": "hailo10",
        },
        {
            "id": "deepx_m1_full", "type": "same_backend_reference",
            "full": "deepx_m1", "stage1": "deepx_m1",
            "stage2": "deepx_m1",
        },
    ]
    base = {
        "run_profiles": run_profiles,
        "quality_gate": {
            "statistics": {"execution_location": "central_management"},
        },
        "hardware_targets": [
            {
                "id": "orin_nx_hailo8_01",
                "accelerator": "hailo8",
                "enabled": True,
            },
            {
                "id": "orin_nx_hailo10_01",
                "accelerator": "hailo10h",
                "enabled": True,
            },
            {
                "id": "orin_nx_deepx_m1_01",
                "accelerator": "deepx_m1",
                "enabled": True,
            },
        ],
        "model_suite": {"primary": [
            {"id": "resnet50", "task": "classification"},
            {"id": "yolo26s", "task": "detection"},
            {"id": "yolov7_paper", "task": "detection"},
        ]},
        "native_producers": {"enabled": True},
    }

    def run_profile(payload: dict[str, object]) -> subprocess.CompletedProcess[str]:
        path = tmp_path / "profile.yaml"
        # JSON is a strict YAML subset and avoids introducing a test-side dumper.
        path.write_text(json.dumps(payload), encoding="utf-8")
        return subprocess.run(
            [sys.executable, "-B", "-", str(path)],
            input=profile_code,
            text=True,
            capture_output=True,
            check=False,
            cwd=ROOT,
        )

    valid = run_profile(base)
    assert valid.returncode == 0, valid.stderr
    assert "PASS Profilvertrag" in valid.stdout

    extra_mixed = json.loads(json.dumps(base))
    extra_mixed["run_profiles"].append({
        "id": "tensorrt_to_hailo8", "type": "mixed_backend",
        "stage1": "tensorrt", "stage2": "hailo8",
    })
    with_targets = json.loads(json.dumps(base))
    with_targets["targets"] = []
    missing_full = json.loads(json.dumps(base))
    missing_full["run_profiles"][1]["enabled"] = False
    wrong_semantics = json.loads(json.dumps(base))
    wrong_semantics["run_profiles"][2]["stage2"] = "tensorrt"
    extra_model = json.loads(json.dumps(base))
    extra_model["model_suite"]["primary"].append({
        "id": "mobilenet_v3", "task": "classification",
    })

    for invalid in (
        extra_mixed, with_targets, missing_full, wrong_semantics, extra_model,
    ):
        result = run_profile(invalid)
        assert result.returncode != 0
        assert "PASS Profilvertrag" not in result.stdout


def test_v27518_local_harness_covers_native_evidence_closure() -> None:
    script = (ROOT / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    assert script.count("tests/test_v2740_native_execution_recovery.py") == 2
    assert script.count("tests/test_v2752_small_acceptance.py") == 2
    assert "scripts/run_v27521_cache_canary.sh" in script
    assert "scripts/run_v27521_small_acceptance.sh" in script
    assert "scripts/run_v27521_full_only_check.sh" in script


def test_v27518_small_and_fresh_acceptance_cover_new_field_repairs() -> None:
    small = (ROOT / "scripts/run_v27521_small_acceptance.sh").read_text(
        encoding="utf-8"
    )
    local = (ROOT / "scripts/run_local_acceptance.sh").read_text(
        encoding="utf-8"
    )
    for target in (
        "tests/test_v27516_read_only_run_admission.py",
        "tests/test_v27516_primary_remote_error_cache_policy.py",
        "tests/test_v269e_start_snapshot_backfill.py",
        "tests/test_v2712_energy_resume.py",
        "tests/test_v27515_deepx_source_binding.py",
        "tests/test_v27515_deepx_reporter_context.py",
        "tests/test_v27515_hailo_vendor_repairs.py",
        "tests/test_energy_final_contract_regressions.py",
        "tests/test_v2711_direct_bn6_producer_binding.py",
        "tests/test_v27518_writer_and_debug_pack_closure.py",
        "tests/test_v27518_deepx_sealed_chain.py",
        "tests/test_v270i_native_full_validation_p0.py",
    ):
        assert target in small
        # Once in the source-tree focus block and once in Fresh Extract.
        assert local.count(target) == 2
    assert "smoke_hailo10_hef_runner.py" in small
    assert "scripts/run_v27514_vendor_full_check.sh" not in small
    assert not (ROOT / "scripts/run_v27515_cache_canary.sh").exists()
    assert not (ROOT / "scripts/run_v27515_small_acceptance.sh").exists()
    assert not (ROOT / "scripts/run_v27515_vendor_full_check.sh").exists()
    assert "build_source_manifest.py --root" not in small


def test_v27516_vendor_wrapper_resolves_exactly_one_fresh_manifest(
    tmp_path: Path,
) -> None:
    script = (
        ROOT / "scripts/run_v27521_full_only_check.sh"
    ).read_text(encoding="utf-8")
    match = re.search(
        r"<<'PYTHON'\n(?P<code>.*?)\nPYTHON\n\)\"",
        script,
        flags=re.DOTALL,
    )
    assert match is not None
    discovery_code = match.group("code")

    runs_root = tmp_path / "runs"
    runs_root.mkdir()
    profile = tmp_path / "profile.yaml"
    profile.write_text("profile: v27516-test\n", encoding="utf-8")

    def write_run(
        name: str,
        *,
        status: str = "ok",
        summary_status: str | None = "ok",
        malformed: bool = False,
    ) -> Path:
        run_dir = runs_root / name
        run_dir.mkdir()
        manifest = run_dir / "run_manifest.json"
        if malformed:
            manifest.write_text("{not-json", encoding="utf-8")
            return run_dir
        manifest.write_text(json.dumps({
                "schema": "onnx-splitpoint/evaluation-run-manifest",
                "run_id": name,
                "run_dir": str(run_dir),
                "profile_path": str(profile),
                "status": status,
                "current_tool_version": HISTORICAL_FULL_ONLY_VERSION,
                "current_workflow_version": HISTORICAL_FULL_ONLY_BUILD,
                "options": {
                    "require_fresh_run": True,
                    "execution_mode": "generate_and_run",
                },
            }), encoding="utf-8")
        if summary_status is not None:
            reports = run_dir / "reports"
            reports.mkdir()
            (reports / "run_status_summary.json").write_text(
                json.dumps({
                    "schema": "onnx-splitpoint/run-status-summary",
                    "run_id": name,
                    "status": summary_status,
                }),
                encoding="utf-8",
            )
        return run_dir

    def set_matching_status(run_dir: Path, status: str) -> None:
        manifest_path = run_dir / "run_manifest.json"
        summary_path = run_dir / "reports" / "run_status_summary.json"
        manifest_payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        manifest_payload["status"] = status
        summary_payload["status"] = status
        manifest_path.write_text(json.dumps(manifest_payload), encoding="utf-8")
        summary_path.write_text(json.dumps(summary_payload), encoding="utf-8")

    preexisting = write_run("preexisting")
    snapshot_path = runs_root / ".snapshot"
    snapshot_path.write_text("", encoding="utf-8")
    snapshot_children = []
    for child in sorted(runs_root.iterdir(), key=lambda path: path.name):
        child_stat = child.lstat()
        snapshot_children.append({
            "name": child.name,
            "device": int(child_stat.st_dev),
            "inode": int(child_stat.st_ino),
            "is_directory": child.is_dir() and not child.is_symlink(),
            "is_symlink": child.is_symlink(),
        })
    snapshot_path.write_text(json.dumps({
        "schema": "onnx-splitpoint/full-only-run-directory-snapshot",
        "schema_version": 1,
        "root": str(runs_root),
        "children": snapshot_children,
    }), encoding="utf-8")
    marker = runs_root / ".marker"
    marker.write_text("", encoding="utf-8")
    os.utime(marker, ns=(1, 1))
    status_result = runs_root / ".status-result"
    status_result.write_text("", encoding="utf-8")
    # Its manifest is newer than the marker, but its directory identity existed
    # before launch and must therefore remain ineligible.
    os.utime(preexisting / "run_manifest.json", None)

    expected = write_run("fresh_one")
    write_run("newer_running", status="running", summary_status="running")
    write_run("newer_malformed", malformed=True)
    write_run("newer_missing_summary", summary_status=None)
    command = [
        sys.executable, "-B", "-", str(marker), str(runs_root), str(profile),
        str(snapshot_path), "0", str(status_result),
    ]
    one = subprocess.run(
        command, input=discovery_code, text=True, capture_output=True,
        check=False,
    )
    assert one.returncode == 0, one.stderr
    assert Path(one.stdout.strip()) == expected
    assert status_result.read_text(encoding="utf-8").strip() == "ok"

    write_run("fresh_two")
    multiple = subprocess.run(
        command, input=discovery_code, text=True, capture_output=True,
        check=False,
    )
    assert multiple.returncode != 0
    assert "Genau ein neuer vollständiger EvaluationRun" in multiple.stderr

    # A controlled partial may still contain the exact Full-only matrix. Admit
    # it to the strict scope-local checker, but never admit a failed workflow.
    fresh_two = runs_root / "fresh_two"
    set_matching_status(fresh_two, "running")
    set_matching_status(expected, "partial")
    partial = subprocess.run(
        command, input=discovery_code, text=True, capture_output=True,
        check=False,
    )
    assert partial.returncode == 0, partial.stderr
    assert Path(partial.stdout.strip()) == expected
    assert status_result.read_text(encoding="utf-8").strip() == "partial"

    checker = subprocess.run(
        [
            sys.executable,
            "-B",
            str(ROOT / "scripts" / "verify_native_full_rows.py"),
            "--run-dir",
            str(expected),
            "--scope",
            "vendor",
            "--format",
            "table",
        ],
        text=True,
        capture_output=True,
        check=False,
    )
    assert checker.returncode in {2, 3}
    assert "PASS Vendor-Full passed=9/9" not in checker.stdout + checker.stderr

    set_matching_status(expected, "failed")
    failed = subprocess.run(
        command, input=discovery_code, text=True, capture_output=True,
        check=False,
    )
    assert failed.returncode != 0
    assert "gefunden=0" in failed.stderr

    diagnostic = subprocess.run(
        command[:-2] + ["1", str(status_result)],
        input=discovery_code,
        text=True,
        capture_output=True,
        check=False,
    )
    assert diagnostic.returncode == 0, diagnostic.stderr
    assert Path(diagnostic.stdout.strip()) == expected
    assert status_result.read_text(encoding="utf-8").strip() == "failed"

    diagnostic_fatal = subprocess.run(
        command[:-2] + ["2", str(status_result)],
        input=discovery_code,
        text=True,
        capture_output=True,
        check=False,
    )
    assert diagnostic_fatal.returncode == 0, diagnostic_fatal.stderr
    assert Path(diagnostic_fatal.stdout.strip()) == expected

    # Nonterminal/cancelled runs and the malformed/missing-summary fixtures
    # above are ignored before the strict checker is invoked.
    set_matching_status(expected, "cancelled")
    none = subprocess.run(
        command, input=discovery_code, text=True, capture_output=True,
        check=False,
    )
    assert none.returncode != 0
    assert "gefunden=0" in none.stderr


def test_v275_preprocessing_paths_are_in_critical_build_inventory() -> None:
    critical = set(package_build_snapshot().get("critical_module_sha256") or {})
    assert {
        "preprocessing_contract.py",
        "v60m_policy.py",
        "benchmark/model_preparation.py",
        "gui_app.py",
        "hailo_backend.py",
        "workflow/benchmark_binding.py",
        "workflow/runner.py",
        "filesystem_admission.py",
        "workflow/analysis_pack.py",
        "workflow/debug_pack_policy.py",
        "workflow/run_discovery.py",
        "runners/native_full_input.py",
        "native_execution_contract.py",
        "remote_runtime_closure.py",
        "hailo_full_contract_promotion.py",
        "resources/remote_scripts/materialize_cache_verify_native_split_binding.py",
    } <= critical
