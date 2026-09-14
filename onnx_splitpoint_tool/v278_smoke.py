"""Compact, hardware-independent release smoke for version 2.78.4."""
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
from .v277_smoke import REQUIRED_FEATURES as V277_REQUIRED_FEATURES


VERSION = "2.78.4"
LINEAGE = "v2.78"
BUILD_ID = "v2.78.4-gate-a-planned-stop-audit-launch"
NEW_FEATURES = {
    "gate_a_planned_stop_audit_launch",
    "forced_audit_deployment_anchor_first",
    "frozen_seven_model_b500_audit20_launch",
    "frozen_hardware_registry_projection",
    "real_request_reference_descriptor_compatibility",
    "immutable_management_cpu_reference_snapshots",
    "read_only_existing_evidence_verification",
    "self_reference_report_scope_fix",
    "exact_hailo_build_evidence_recovery_index",
    "separate_runtime_evidence_ledger",
    "hailo8_first_common_anchor_feasibility",
    "hailo_feasibility_budget_terminal_stop",
    "generation_state_crash_safe_cold_reservation",
    "packaged_yolo11_b5_gate_a_profile",
    "generic_hailo_native_quality_selection_gate",
    "deepx_model_sidecar_identity",
    "resnet_quantized_interface_semantic_aggregation",
    "classification_candidate_only_tristate",
}
REQUIRED_FEATURES = set(V277_REQUIRED_FEATURES) | NEW_FEATURES


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    workflow_source = (
        root / "onnx_splitpoint_tool/workflow/runner.py"
    ).read_text(encoding="utf-8")
    required_files = (
        "onnx_splitpoint_tool/v278_smoke.py",
        "onnx_splitpoint_tool/existing_evidence_verifier.py",
        "onnx_splitpoint_tool/management_reference.py",
        "onnx_splitpoint_tool/quality_replay.py",
        "onnx_splitpoint_tool/quality_service.py",
        "onnx_splitpoint_tool/build_evidence.py",
        "onnx_splitpoint_tool/runtime_evidence.py",
        "onnx_splitpoint_tool/v277_smoke.py",
        "onnx_splitpoint_tool/backend_semantic_smoke.py",
        "onnx_splitpoint_tool/ranking_methods.py",
        "onnx_splitpoint_tool/workflow/results.py",
        "onnx_splitpoint_tool/workflow/cross_runner_reporting.py",
        "onnx_splitpoint_tool/workflow/missing_full_quality_attestation.py",
        "scripts/run_v278_small_acceptance.sh",
        "scripts/verify_existing_adapter_evidence.py",
        "scripts/recover_b5_build_evidence.py",
        "scripts/run_v2783_yolo11_hailo8_first_gate.sh",
        "scripts/verify_v2783_yolo11_gate_a_output.py",
        "profiles/yolo11l_v2783_hailo8_first_b5_gate_a.yaml",
        "scripts/run_v2784_seven_model_long_overnight.sh",
        "profiles/complete_set_7models_v2784_b500_audit20.yaml",
        "scripts/run_v277_small_acceptance.sh",
        "scripts/native_yolo_full_self_reference_probe.py",
        "onnx_splitpoint_tool/resources/remote_scripts/native_yolo_full_self_reference_probe.py",
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
        "tests/test_v278_release_provenance.py",
        "tests/test_v2781_existing_evidence_verifier.py",
        "tests/test_v2782_management_reference_resume.py",
        "tests/test_v2782_quality_replay_reference_path.py",
        "tests/test_v2783_build_evidence.py",
        "tests/test_v2783_runtime_evidence.py",
        "tests/test_v2783_hailo8_first_feasibility.py",
        "tests/test_v2783_hailo8_evidence_binding.py",
        "tests/test_v2783_generation_state_atomic.py",
        "tests/test_v2783_deepx_model_manifest.py",
        "tests/test_v2783_resnet_quantized_interface_aggregation.py",
        "tests/test_v2783_yolo11_gate_a_profile.py",
        "tests/test_v2783_yolo11_hailo8_first_launcher.py",
        "tests/test_v2783_yolo11_gate_a_output_verifier.py",
        "tests/v2783_gate_a_fixture_builder.py",
        "tests/test_v2784_seven_model_long_profile.py",
        "tests/test_v2784_seven_model_overnight_launcher.py",
        "tests/test_v27519_management_cpu_production_chain.py",
        "tests/test_v277_release_provenance.py",
        "tests/test_completed_attestation_contract_repairs.py",
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
                'version = "2.78.4"',
                (
                    "onnx-splitpoint-smoke-v278 = "
                    '"onnx_splitpoint_tool.v278_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-78 = "
                    '"onnx_splitpoint_tool.v278_smoke:main"'
                ),
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
        print("FAIL v2.78 smoke: " + ", ".join(failed))
        return 1
    print("PASS v2.78 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
