"""Compact, hardware-independent release smoke for version 2.76.2."""
from __future__ import annotations

from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v27550_smoke import REQUIRED_FEATURES as V27550_REQUIRED_FEATURES


VERSION = "2.76.2"
LINEAGE = "v2.76"
BUILD_ID = "v2.76.2-normalized-cross-runner-alias-closure"
NEW_FEATURES = {
    "read_only_scientific_reprojection",
    "separate_reprojection_output_root",
    "endpoint_lifecycle_evidence_ledger",
    "explicit_terminal_endpoint_reasons",
    "nonzero_runner_rc_fail_closed",
    "numeric_input_identity_projection",
    "native_disabled_status_projection",
    "analysis_pack_observation_surfaces",
    "historical_runner_rc_log_reclassification",
    "external_native_replay_source_binding",
    "ranking_sensitivity_cohort_surfaces",
    "cross_runner_cohort_top1_metrics",
    "backend_partial_artifact_retention",
    "parallel_builder_exception_partial_retention",
    "hailo10h_physical_alias_canonicalization",
    "post_build_minimum_status",
    "yolo26_base_conv_resolution_repair",
    "yolo26_exact_base_conv_retry",
    "frozen_native_intersection_cross_runner_join",
    "exact_generic_request_identity_reprojection",
    "cross_runner_technical_micro_backend_aggregates",
    "development_actual_strata_leader",
    "native_replay_final_path_rebase",
    "normalized_generic_metric_alias_projection",
    "cross_runner_markdown_identity_diagnostics",
    "native_scientific_claim_counter_semantics",
}
REQUIRED_FEATURES = set(V27550_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    workflow_source = (
        root / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    required_files = (
        "onnx_splitpoint_tool/v276_smoke.py",
        "onnx_splitpoint_tool/workflow/scientific_replay.py",
        "onnx_splitpoint_tool/workflow/scientific_reporting.py",
        "onnx_splitpoint_tool/workflow/endpoint_lifecycle.py",
        "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
        "onnx_splitpoint_tool/reporting_quality_decomposition.py",
        "onnx_splitpoint_tool/resources/templates/benchmark_suite.py.txt",
        "scripts/replay_scientific_reports.py",
        "scripts/run_v276_small_acceptance.sh",
        "scripts/replay_native_supplement_reports.py",
        "tests/test_v276_release_provenance.py",
        "tests/test_v276_endpoint_lifecycle.py",
        "tests/test_v276_pipeline_repairs.py",
        "tests/test_v276_quality_projection.py",
        "tests/test_v276_reporting_identity.py",
        "tests/test_v276_native_supplement_replay.py",
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
        "files": all((root / name).is_file() for name in required_files),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.76.2"',
                (
                    "onnx-splitpoint-smoke-v276 = "
                    '"onnx_splitpoint_tool.v276_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-76 = "
                    '"onnx_splitpoint_tool.v276_smoke:main"'
                ),
            )
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL v2.76 smoke: " + ", ".join(failed))
        return 1
    print("PASS v2.76 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
