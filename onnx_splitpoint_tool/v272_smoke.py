"""Hardware-independent release-contract smoke test for version 2.75.40."""
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
    "compact_canonical_debug_pack",
    "full_workflow_log_debug_pack",
    "native_ranking_observation_ingestion",
    "simplified_model_usage_profile_editor",
    "score_independent_profile_schema",
    "completed_detection_execution_contract",
    "completed_detection_hotloop_tail",
    "semantic_yolov7_shape_stride_mapping",
    "separate_detection_hash_domains",
    "same_hotloop_and_independent_replay_relation",
    "exact_quality_evidence_binding_axes",
    "planned_vs_matrix_energy_coverage",
    "final_all_split_energy_contract",
    "hailo10_yolo26_numeric_drift_fixture",
    "canonical_completed_result_artifact_transport_hash",
    "contract_bound_yolov7_activation_strategy",
    "detection_completion_preflight_identity",
    "prospective_hailo10_yolo26_claim_exclusion",
    "four_dimensional_claim_exclusion_reporting",
    "hailo8_system_trt_process_local_hailo_sites",
    "mixed_runtime_import_preflight",
    "primary_child_error_precedence",
    "completed_v2_structural_consumer_precedence",
    "orthogonal_quality_observation_and_accuracy",
    "quality_first_energy_collector_admission",
    "contract_frozen_energy_repeat_counts",
    "complete_energy_exclusion_ledger",
    "canonical_energy_coverage_projection",
    "post_native_expected_energy_matrix",
    "shared_sealed_energy_quality_admission",
    "fail_closed_energy_plan_result_ledger",
    "completed_v2_prelegacy_consumer_normalization",
    "legacy_positive_energy_evidence_projection",
    "canonical_profile_selection_snapshot_identity",
    "setup_scoped_central_quality_request_identity",
    "semantic_central_quality_mirror_deduplication",
    "portable_native_split_semantic_dumps",
    "hailo_full_raw_endpoint_quality_binding",
    "timed_frozen_decoder_quality_invariant",
    "smoke_bound_negative_quality_observation_energy",
    "canonical_detection_quality_contract_projection",
    "manifest_bound_central_quality_staging_deduplication",
    "backend_bound_resume_artifact_roles",
    "hailo8_exact_row_resume_contract",
    "hailo8_python_detection_preflight_source_recovery",
    "native_energy_runtime_success_admission",
    "deepx_exact_row_resume_artifact_rehydration",
    "evaluated_matrix_claim_scope",
    "optional_ranking_generalization_scope",
    "scope_conditional_readiness",
    "canonical_confirmatory_holdout_runtime",
    "semantic_trt_cache_elapsed_exclusion",
    "yolov7_final_contract_canary",
    "campaign_preflight_success_exit_contract",
    "exact_yolov7_paper_final_template",
    "runtime_exact_campaign_contract_templates",
    "yolov7_paper_final_canary",
    "inherited_energy_method_runtime_binding_gate",
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
                "onnx-splitpoint-smoke-v272 = "
                '"onnx_splitpoint_tool.v272_smoke:main"'
            ),
            (
                "onnx-splitpoint-smoke-v2-72 = "
                '"onnx_splitpoint_tool.v272_smoke:main"'
            ),
        )
    )


def _release_builder_contract_present(project_root: Path) -> bool:
    path = project_root / "scripts" / "build_source_release.py"
    if not path.is_file():
        return False
    source = path.read_text(encoding="utf-8")
    return all(
        marker in source
        for marker in (
            "SOURCE_MANIFEST.json",
            "SHA256SUMS.txt",
            "ZIP_DEFLATED",
            "compresslevel=9",
            "1980, 1, 1, 0, 0, 0",
            "verify_archive",
        )
    )


def main() -> int:
    project_root = Path(__file__).resolve().parents[1]
    build = package_build_snapshot()
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
        "critical_module_set_complete": (
            build.get("critical_module_set_complete") is True
            and {
                "v272_smoke.py",
                "resume_hailo8_source_recovery.py",
            }.issubset(
                set(build.get("critical_module_sha256") or {})
            )
        ),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
        "release_entry_points": _release_entry_points_present(project_root),
        "source_release_builder": _release_builder_contract_present(
            project_root,
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v272-smoke",
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
