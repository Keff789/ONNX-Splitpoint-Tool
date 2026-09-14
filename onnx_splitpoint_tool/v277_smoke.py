"""Compact, hardware-independent release smoke for version 2.77.15."""
from __future__ import annotations

from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .ranking_methods import (
    RANKING_METHOD_IMPLEMENTATION,
    WORKFLOW_RANKING_METHOD,
)
from .v276_smoke import REQUIRED_FEATURES as V276_REQUIRED_FEATURES


VERSION = "2.77.15"
LINEAGE = "v2.77"
BUILD_ID = "v2.77.15-yolo11-completed-v2-reference-projection"
NEW_FEATURES = {
    "yolo11_completed_v2_ultralytics_reference_projection",
    "backend_semantic_smoke_spec_v1",
    "prepared_rgb_exact_identity_smoke",
    "onnx_hailo_fixup_cpu_parity_smoke",
    "yolo_raw_head_host_tail_parity_smoke",
    "optional_hailo8_structural_artifact_runtime_canary",
    "manifest_bound_remote_canary_payload",
    "leased_bounded_remote_canary",
    "fd_bound_regular_payload_member_materialization",
    "hailo8_multimodel_hardlink_payload_closure",
    "hailo8_canary_compact_stdout_full_receipt",
    "full_only_quality_hardware_smoke_not_applicable",
    "full_only_quality_offline_status_projection",
    "quality_reference_decision_projection",
    "bootstrap_bound_evidence_reporting",
    "missing_full_quality_yolo11_sha256_normalization",
    "missing_full_quality_targeted_rebuild_precedence",
    "missing_full_quality_exact_complete_audit",
    "hailo_full_quality_timed_output_order_reprojection",
    "decoded_pre_nms_full_output_contract",
    "missing_full_quality_exact_resume",
    "missing_full_quality_fresh_start_guard_override",
    "missing_full_quality_transport_run_isolation",
    "missing_full_quality_downstream_supersession_attestation",
    "missing_full_quality_cohort_reuse_preflight",
    "missing_full_quality_abandoned_v2779_recovery",
    "missing_full_quality_terminal_session_persistence",
    "missing_full_quality_dynamic_preservation_counts",
    "missing_full_quality_workflow_status_fail_closed",
    "missing_full_quality_must_reuse_stage_guard",
    "missing_full_quality_attempt_transport_isolation",
    "missing_full_quality_zero_byte_artifact_reuse",
    "missing_full_quality_historical_success_stderr_attestation",
    "remote_success_stderr_materialization",
    "downstream_formal_contract_artifact_registration",
    "runtime_finalization_alias_artifact_registration",
    "central_quality_result_preservation",
    "phase5_physical_backend_projection",
    "explicit_empty_hailo_target_authority",
    "yolo11_dfl16_native_full_adapter",
    "v2776_frozen_detection_contract_compatibility",
    "forced_case_exact_generator_scope",
    "managed_venv_subprocess_path_projection",
    "managed_venv_symlink_safe_path_projection",
    "optional_onnxsim_recovery_diagnostic",
    "automatic_parser_path_before_endpoint_retry",
    "dfc_visible_tool_identity_endpoint_projection",
    "dfc_output_slot_exact_projection",
    "archived_endpoint_graph_attestation",
    "archived_part1_sha256_binding",
    "case_bound_base_conv_retry",
    "negative_compiler_capability_evidence_closure",
    "hailo_full_not_requested_accounting_closure",
    "cut_bytes_only_ranking_freeze",
    "deterministic_boundary_case_tiebreak",
    "candidate_generation_policy_preserved",
    "cross_runner_endpoint_attestation_provenance_normalization",
    "cross_runner_archived_normalized_default_migration",
    "native_replay_quality_gated_combined_summary",
    "hailo_build_full_false_authoritative_skip",
    "hailo_full_policy_three_layer_propagation",
    "hailo_matrix_full_variant_scope_guard",
    "part1_pair_build_fail_closed_canary",
    "self_contained_canary_evidence_manifest",
}
REQUIRED_FEATURES = set(V276_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    workflow_source = (
        root / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    required_files = (
        "onnx_splitpoint_tool/v277_smoke.py",
        "onnx_splitpoint_tool/backend_semantic_smoke.py",
        "onnx_splitpoint_tool/ranking_methods.py",
        "onnx_splitpoint_tool/workflow/results.py",
        "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
        "onnx_splitpoint_tool/workflow/missing_full_quality_attestation.py",
        "scripts/run_v277_small_acceptance.sh",
        "scripts/run_backend_semantic_smokes.py",
        "scripts/native_producer_validate_visualize.py",
        "scripts/run_v27713_hailo8_artifact_canary.py",
        "scripts/verify_v27713_hailo8_canary_payload.py",
        "scripts/replay_scientific_reports.py",
        "V2772_HAILO_PARALLEL_CANARY_README.md",
        "profiles/resnet50_v2772_hailo_parallel_build_canary.yaml",
        "scripts/preflight_v2772_hailo_parallel_build_canary.py",
        "scripts/run_v2772_hailo_parallel_build_canary.sh",
        "scripts/verify_v2772_hailo_parallel_build_canary.py",
        "scripts/probe_v2776_yolo26_archived_part1.py",
        "scripts/resume_missing_full_quality.py",
        "tests/test_v277_release_provenance.py",
        "tests/test_v270l_completed_endpoint_v2.py",
        "tests/test_v27713_backend_semantic_smoke.py",
        "tests/test_v27713_hailo8_artifact_canary.py",
        "tests/test_v277_cut_bytes_workflow_freeze.py",
        "tests/test_v2772_hailo_full_scope.py",
        "tests/test_v2772_hailo_parallel_canary_profile.py",
        "tests/test_v2772_hailo_parallel_canary_verifier.py",
        "tests/test_v2776_direct_parse_probe.py",
        "tests/test_v2777_phase5_hailo_projection.py",
        "tests/test_v2777_yolo11_native_adapter.py",
        "tests/test_v2778_missing_full_quality_resume.py",
        "tests/test_missing_full_quality_attestation.py",
        "tests/test_resume_missing_full_quality_wrapper_hardening.py",
        "tests/test_v2779_targeted_resume_attempt_isolation.py",
        "tests/test_v27710_abandoned_resume_recovery.py",
        "tests/test_v27710_reuse_cohort_attestation.py",
        "tests/test_deepx_full_only_quality_dispatch.py",
        "tests/test_v27539_remote_dispatch_propagation.py",
        "tests/test_v276_cross_runner_reporting.py",
    )
    checks = {
        "version": __version__ == VERSION,
        "release": __release__ == VERSION,
        "lineage": __development_lineage__ == LINEAGE,
        "build": (
            __build_id__ == BUILD_ID
            and f'WORKFLOW_VERSION = "{BUILD_ID}"' in workflow_source
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "ranker": (
            WORKFLOW_RANKING_METHOD == "cut_bytes_only"
            and RANKING_METHOD_IMPLEMENTATION
            == "v277-cut-bytes-only-workflow-freeze-1"
        ),
        "files": all((root / name).is_file() for name in required_files),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.77.15"',
                (
                    "onnx-splitpoint-smoke-v277 = "
                    '"onnx_splitpoint_tool.v277_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-77 = "
                    '"onnx_splitpoint_tool.v277_smoke:main"'
                ),
                (
                    "onnx-splitpoint-backend-semantic-smoke = "
                    '"onnx_splitpoint_tool.backend_semantic_smoke:main"'
                ),
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL v2.77 smoke: " + ", ".join(failed))
        return 1
    print("PASS v2.77 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
