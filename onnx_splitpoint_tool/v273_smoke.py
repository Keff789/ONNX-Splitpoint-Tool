"""Hardware-independent release-contract smoke for the 2.73 release family."""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v269a_smoke import _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.75.40"
BUILD_ID = "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair"
CURRENT_VERSION = "2.75.47"
CURRENT_BUILD_ID = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
REQUIRED_FEATURES = {
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
    "central_trt_logical_signed_backend_alias",
    "native_detection_mean_iou_085",
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
    "deepx_full_sealed_input_contract_projection",
    "collected_native_full_semantic_path_rebase",
    "terminal_full_semantic_fail_closed",
    "native_runtime_policy_alias_projection",
    "gui_offline_replay_core_preservation",
    "vendor_runtime_policy_hash_diagnostic_only",
    "hailo8_yolov7_canonical_raw_head_loader_chain",
    "purpose_bound_evaluation_run_discovery",
    "read_only_pack_source_contract",
    "structured_partial_resume_admission",
    "nonmutating_local_remote_storage_preflight",
    "cold_warm_remote_capacity_contract",
    "owned_managed_trt_cache_retention_contract",
    "same_host_remote_storage_serialization",
    "terminal_storage_primary_error_preservation",
    "hailo_full_management_free_remote_closure",
    "runtime_plan_post_normalization_finalization",
    "three_node_native_contract_preflight",
    "failure_safe_full_only_wrapper_reporting",
    "vendor_full_nine_row_acceptance",
    "portable_deepx_full_semantic_manifest",
    "sealed_deepx_quality_model_identity",
    "idempotent_hailo_full_contract_promotion",
    "canonical_quality_record_endpoint_contract",
    "native_energy_runtime_success_admission",
    "classification_inclusive_margin_exact_counts",
    "classification_boundary_ulp_fallback",
    "cross_host_dataset_content_identity",
    "native_full_completed_task_quality_authority",
    "completed_hotloop_semantic_primary",
    "completed_full_sentinel_artifact_binding",
    "negative_quality_decision_completion_semantics",
    "unavailable_evidence_partial_status",
    "isolated_test_home",
    "deterministic_clean_source_allowlist",
    "shell_safe_local_acceptance",
    "explicit_current_test_discovery",
    "verify_only_source_manifest_acceptance",
    "backend_bound_native_full_restage_roles",
    "shared_profile_snapshot_workflow_options",
    "fresh_standard_execution_path_launcher",
    "withdrawn_parallel_acceptance_stack_removed",
    "exclusive_evaluation_run_writer_lock",
    "frozen_resume_profile_plan_contract",
    "recursive_process_tree_cancellation",
    "atomic_stage_result_checkpoints",
    "atomic_native_energy_row_journal",
    "resumable_native_energy_row_journal",
    "terminal_native_performance_handoff",
    "native_energy_planner_invariant",
    "explicit_hailo8_setup_identity",
    "shared_native_remote_runtime_closure",
    "plan_bound_native_split_setup_identity",
    "canonical_native_energy_failure_ledger",
    "split_energy_engine_artifact_projection",
    "completed_nms_central_quality_projection",
    "cache_verify_only_artifact_policy",
    "cache_verify_only_exact_plan_attestation",
    "cache_verify_only_compiler_dispatch_fence",
    "cache_verify_only_canary_profile",
    "profile_driven_cache_canary_run_id",
    "cache_canary_complete_remote_runtime_closure",
    "attested_smoke_cache_identity",
    "cross_suite_exact_cache_resolution",
    "invalid_cache_binding_not_identity_miss",
    "mixed_runtime_separate_hailo_interpreter",
    "receipt_signed_hailo_compiler_sibling",
    "deepx_prepared_input_exact_transport",
    "same_input_detection_reference_gate",
    "completed_v2_persistence_projection",
    "dependency_free_native_split_cache_replay",
    "holdout_adapter_source_freeze",
    "diagnostic_only_dequant_sweep",
    "canonical_native_result_verification",
}


def _release_entry_points_present(project_root: Path) -> bool:
    pyproject = (project_root / "pyproject.toml").read_text(encoding="utf-8")
    version_present = any(
        marker in pyproject
        for marker in (
            'version = "2.75.40"',
            'version = "2.75.47"',
        )
    )
    return version_present and all(
        marker in pyproject
        for marker in (
            (
                "onnx-splitpoint-smoke-v273 = "
                '"onnx_splitpoint_tool.v273_smoke:main"'
            ),
            (
                "onnx-splitpoint-smoke-v2-73 = "
                '"onnx_splitpoint_tool.v273_smoke:main"'
            ),
        )
    )


def _current_release_documents_present(project_root: Path) -> bool:
    version = str(__version__)
    return all(
        (project_root / name).is_file()
        for name in (
            f"TESTANLEITUNG_{version}.md",
            f"VERSION_{version}_BUILD_AND_TEST_REPORT.md",
        )
    )


def _clean_release_builder_present(project_root: Path) -> bool:
    manifest_builder = (
        project_root / "scripts" / "build_source_manifest.py"
    ).read_text(encoding="utf-8")
    release_builder = (
        project_root / "scripts" / "build_source_release.py"
    ).read_text(encoding="utf-8")
    return all(
        marker in manifest_builder
        for marker in (
            "ALLOWED_ROOT_FILES",
            "ALLOWED_SOURCE_TREES",
            "ALLOWED_DOC_FILES",
            "_allowlisted",
        )
    ) and all(
        marker in release_builder
        for marker in (
            "SOURCE_MANIFEST.json",
            "SHA256SUMS.txt",
            "FIXED_ZIP_TIMESTAMP",
            "verify_archive",
        )
    )


def _local_acceptance_contract(project_root: Path) -> dict[str, bool]:
    pyproject = (project_root / "pyproject.toml").read_text(
        encoding="utf-8",
    )
    script_path = project_root / "scripts" / "run_local_acceptance.sh"
    if not script_path.is_file():
        return {
            "test_discovery": False,
            "shell_safe": False,
            "manifest_verify_only": False,
        }
    source = script_path.read_text(encoding="utf-8")
    manifest_invocations = source.split(
        '"$PY" -B scripts/build_source_manifest.py'
    )[1:]
    return {
        "test_discovery": (
            'testpaths = ["tests"]' in pyproject
            and '"8/10 - Complete suite vs frozen v2.73.8 failure baseline" tests'
            in source
        ),
        "shell_safe": (
            bool(script_path.stat().st_mode & 0o100)
            and '[[ "${BASH_SOURCE[0]}" != "$0" ]]' in source
            and "exit 0" not in source
        ),
        "manifest_verify_only": (
            len(manifest_invocations) == 3
            and all(
                "--verify" in invocation.split("\n\n", 1)[0]
                for invocation in manifest_invocations
            )
            and "build_source_manifest.py --root" not in source
        ),
    }


def _standard_execution_path_contract(project_root: Path) -> dict[str, bool]:
    launcher = project_root / "scripts" / "run_fresh_standard_workflow.sh"
    resolver = (
        project_root
        / "onnx_splitpoint_tool"
        / "workflow"
        / "profile_options.py"
    )
    obsolete = (
        "scripts/run_v2732_full_hardware_gates.py",
        "scripts/run_v2732_hardware_acceptance.sh",
        "scripts/v2732_hardware_acceptance_verify.py",
        "scripts/v2732_semantic_quality_gate.py",
        "tests/test_v2732_fresh_selected_energy.py",
        "tests/test_v2732_full_hardware_coordinator.py",
        "tests/test_v2732_hardware_acceptance_harness.py",
        "tests/test_v2732_hardware_acceptance_verify.py",
        "tests/test_v2732_semantic_quality_gate.py",
    )
    source = launcher.read_text(encoding="utf-8") if launcher.is_file() else ""
    return {
        "shared_options_resolver": resolver.is_file(),
        "single_product_entry": (
            bool(launcher.stat().st_mode & 0o100)
            if launcher.is_file()
            else False
        )
        and source.count(
            "onnx_splitpoint_tool.workflow.run_evaluation"
        )
        == 1
        and 'exec "${command[@]}"' in source,
        "fresh_standard_guards": all(
            marker in source
            for marker in (
                "--profile-driven",
                "--require-run-mode standard",
                "--require-fresh-run",
                "--execution-mode generate_and_run",
            )
        ),
        "parallel_stack_absent": not any(
            (project_root / relative).exists() for relative in obsolete
        ),
    }


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    build = package_build_snapshot()
    acceptance = _local_acceptance_contract(project_root)
    standard_path = _standard_execution_path_contract(project_root)
    checks = {
        "version": __version__ in {VERSION, CURRENT_VERSION},
        "release": __release__ in {VERSION, CURRENT_VERSION},
        "lineage": __development_lineage__
        in {f"v{VERSION}", f"v{CURRENT_VERSION}"},
        "workflow": WORKFLOW_VERSION in {BUILD_ID, CURRENT_BUILD_ID},
        "build_id": __build_id__ in {BUILD_ID, CURRENT_BUILD_ID},
        "build_contract": int(__build_contract_version__) == 2,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "source_mirrors": (
            _source_mirrors_match()
            and _all_applicable_remote_source_mirrors_match()
        ),
        "critical_module_set": (
            build.get("critical_module_set_complete") is True
            and int(build.get("critical_module_count") or 0) >= 165
            and {
                "deepx/env_status.py",
                "native_performance_identity.py",
                "preprocessing_contract.py",
                "remote_runtime_closure.py",
                "hailo_full_contract_promotion.py",
                "benchmark/model_preparation.py",
                "gui_app.py",
                "v273_smoke.py",
                "workflow/checkpoints.py",
                "filesystem_admission.py",
                "workflow/analysis_pack.py",
                "workflow/run_discovery.py",
            }.issubset(
                set(build.get("critical_module_sha256") or {})
            )
        ),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
        "release_entry_points": _release_entry_points_present(project_root),
        "current_release_documents": _current_release_documents_present(
            project_root,
        ),
        "clean_source_builder": _clean_release_builder_present(project_root),
        "explicit_test_discovery": acceptance["test_discovery"],
        "shell_safe_local_acceptance": acceptance["shell_safe"],
        "verify_only_source_manifest": acceptance[
            "manifest_verify_only"
        ],
        "shared_profile_options": standard_path["shared_options_resolver"],
        "single_standard_product_entry": standard_path[
            "single_product_entry"
        ],
        "fresh_standard_guards": standard_path["fresh_standard_guards"],
        "withdrawn_parallel_stack_absent": standard_path[
            "parallel_stack_absent"
        ],
    }
    result = {
        "schema": "onnx-splitpoint/v273-smoke",
        "schema_version": 1,
        "ok": all(checks.values()),
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "checks": checks,
        "build": build,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
