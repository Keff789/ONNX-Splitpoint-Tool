"""Hardware-independent release-contract smoke test for version 2.63."""
from __future__ import annotations

import inspect
import json
from types import SimpleNamespace

from . import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __next_major_version__,
    __release__,
    __version__,
)
from .benchmark.services import _benchmark_plan_has_deepx_stage2
from .energy.config import EnergyDefaults, resolve_energy_ab_config
from .management_reference import generate_management_cpu_reference
from .protocol_freeze import (
    CONFIRMATORY_HOLDOUT_ROLE,
    PROTOCOL_MANIFEST_KINDS,
    normalize_evaluation_role,
)
from .quality_service import (
    EVALUATE_PAIRED_QUALITY_UNCERTAINTY,
    GENERATE_CPU_QUALITY_REFERENCE,
    ManagementQualityService,
)
from .workflow.artifacts import package_build_snapshot
from .workflow.execution_binding import (
    _parallel_remote_max_setups,
    _parallel_remote_setups_enabled,
)
from .workflow.runner import (
    WORKFLOW_VERSION,
    EvaluationWorkflowRunner,
    _annotate_candidate_execution_contract_v261e,
    _performance_plan_v263,
)


REQUIRED_FEATURES = {
    "urecs_window_method_ab",
    "screening_window_validation_probe",
    "context_bound_native_full_validation",
    "atomic_verified_debug_pack",
    "auditable_package_build_identity",
    "exact_native_probe_contract_reuse",
    "remote_runtime_sha256_binding",
    "reconnect_safe_window_probe",
    "deepx_3d_hwc_prepared_feed_contract",
    "precision_independent_native_full_pairing",
    "pre_sampling_energy_contract_attestation",
    "dedicated_native_full_energy_hotloops",
    "exact_trtexec_iteration_and_input_binding",
    "split_energy_workload_only_contract",
    "active_duration_matched_energy_pairing",
    "exact_prepared_feed_energy_replay",
    "direct_exact_work_unit_markers",
    "task_bound_preprocess_pairing",
    "task_bound_deepx_compiler_preprocess",
    "canonical_prepared_input_slot_binding",
    "management_node_cpu_quality_reference",
    "management_node_paired_quality_pool",
    "cpu_reference_excluded_from_performance",
    "content_addressed_quality_cache",
    "resource_gated_management_quality",
    "confirmatory_protocol_freeze",
    "energy_same_capture_shadow_ab",
    "deepx_stage2_proxy_request_gate",
    "bounded_parallel_remote_setups",
    "cooperative_process_group_cancellation",
    "prediction_measurement_case_join",
}


def _cpu_reference_is_semantic_only() -> bool:
    plan = _performance_plan_v263(
        {"runs": [{"id": "ort_cpu"}, {"id": "hailo8"}]},
        {"quality_gate": {"execution_location": "central_management"}},
    )
    return (
        [row.get("id") for row in plan.get("runs", [])] == ["hailo8"]
        and plan.get("semantic_reference_run_ids") == ["ort_cpu"]
        and plan.get("performance_excluded_run_ids") == ["ort_cpu"]
        and plan.get("cpu_reference_execution_location") == "central_management"
    )


def _execution_contract_fixes_are_active() -> bool:
    deepx_full = {
        "id": "deepx_m1_full",
        "type": "deepx",
        "variants": ["full"],
        "stage1": {"type": "deepx"},
        "stage2": {"type": "deepx"},
    }
    deepx_stage2 = {
        "id": "tensorrt_to_deepx_m1",
        "type": "matrix",
        "variants": ["split"],
        "stage1": {"type": "onnxruntime", "provider": "tensorrt"},
        "stage2": {"type": "deepx"},
    }
    profile = {"parallel_remote": {"enabled": True, "max_parallel_setups": 3}}
    options = SimpleNamespace(parallel_remote_setups=None, max_parallel_setups=None)
    joined = _annotate_candidate_execution_contract_v261e(
        [
            {"case_id": "b001", "variant": "split"},
            {"case_id": "b002", "variant": "split"},
            {"run_id": "deepx_m1_full", "variant": "full"},
        ],
        {"selected_candidates": [{"case_id": "b001", "rank": 1}]},
    )
    return bool(
        not _benchmark_plan_has_deepx_stage2([deepx_full])
        and _benchmark_plan_has_deepx_stage2([deepx_stage2])
        and _parallel_remote_setups_enabled(options, profile)
        and _parallel_remote_max_setups(options, profile) == 3
        and [row.get("candidate_plan_join_status") for row in joined]
        == ["joined", "candidate_missing", "not_applicable_full_baseline"]
        and callable(getattr(EvaluationWorkflowRunner, "request_cancel", None))
    )


def main() -> int:
    build = package_build_snapshot()
    energy = resolve_energy_ab_config()
    defaults = EnergyDefaults()
    critical_modules = set(build.get("critical_module_sha256", {}))
    checks = {
        "version": __version__ in {"2.63.0", "2.64.0", "2.65.0", "2.66.0", "2.67.0", "2.68.0", "2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "release": __release__ in {"2.63", "2.64", "2.65", "2.66", "2.67", "2.68", "2.69f", "2.70f", "2.70g", "2.70h", "2.70i", "2.70j", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "lineage": __development_lineage__ in {"v2.63", "v2.64", "v2.65", "v2.66", "v2.67", "v2.68", "v2.69f", "v2.70f", "v2.70g", "v2.70h", "v2.70i", "v2.70j", "v2.71.1", "v2.71.2", "v2.71.3", "v2.71.4", "v2.72.0", "v2.72.1", "v2.72.2", "v2.72.3", "v2.72.4", "v2.72.5", "v2.72.6", "v2.72.7", "v2.73.0", "v2.73.6", "v2.73.7", "v2.75.2", "v2.75.3", "v2.75.7", "v2.75.8", "v2.75.9", "v2.75.10", "v2.75.11", "v2.75.12", "v2.75.14", "v2.75.16", "v2.75.17", "v2.75.20", "v2.75.21", "v2.75.22", "v2.75.24", "v2.75.25", "v2.75.26", "v2.75.27", "v2.75.28", "v2.75.30", "v2.75.31", "v2.75.32", "v2.75.38", "v2.75.39", "v2.75.40", "v2.75.41", "v2.75.42", "v2.75.46", "v2.75.47"},
        "next_major": __next_major_version__ == "3.0.0",
        "workflow": WORKFLOW_VERSION in {"v2.63-campaign-ready", "v2.64-campaign-ready", "v2.65-campaign-ready", "v2.66-campaign-ready", "v2.67-campaign-ready", "v2.68-level-playing-field", "v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"},
        "build_id": __build_id__ in {"v2.63.0-campaign-ready", "v2.64.0-campaign-ready", "v2.65.0-campaign-ready", "v2.66.0-campaign-ready", "v2.67.0-campaign-ready", "v2.68.0-level-playing-field", "v2.69.6-hardware-smoke-native-energy-repair", "v2.70.6-legacy-log-callback-repair", "v2.70.7-native-validation-bridge-repair", "v2.70.8-remote-suite-bootstrap-repair", "v2.70.9-native-full-evidence-repair", "v2.70.10-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"},
        "build_contract": __build_contract_version__ == 2,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "protocol_roles": (
            normalize_evaluation_role("holdout") == CONFIRMATORY_HOLDOUT_ROLE
            and normalize_evaluation_role("confirmatory_holdout") == CONFIRMATORY_HOLDOUT_ROLE
            and PROTOCOL_MANIFEST_KINDS == ("candidate", "dag", "prediction", "policy", "energy")
        ),
        "energy_same_capture_shadow_ab": (
            energy.get("valid") is True
            and energy.get("same_raw_capture") is True
            and energy.get("mode") == "shadow"
            and energy.get("auto_switch") is False
            and energy.get("requires_picoscope") is False
            and defaults.window_ab_enabled is True
        ),
        "management_quality_workers": (
            inspect.signature(generate_management_cpu_reference).parameters["workers"].default == 4
            and inspect.signature(ManagementQualityService).parameters["workers"].default == 4
            and GENERATE_CPU_QUALITY_REFERENCE == "generate_cpu_quality_reference"
            and EVALUATE_PAIRED_QUALITY_UNCERTAINTY == "evaluate_paired_quality_uncertainty"
        ),
        "cpu_reference_semantic_only": _cpu_reference_is_semantic_only(),
        "execution_contract_fixes": _execution_contract_fixes_are_active(),
        "critical_module_hashes": (
            len(critical_modules) >= 59
            and all(
                str(value).startswith("sha256:")
                for value in build.get("critical_module_sha256", {}).values()
            )
        ),
        "v263_critical_modules": {
            "benchmark/accuracy_gate.py",
            "benchmark/accuracy_gates.py",
            "benchmark/remote_run.py",
            "benchmark/services.py",
            "energy/config.py",
            "energy/collector.py",
            "deepx/config.py",
            "management_reference.py",
            "native_command_contract.py",
            "native_energy_reporting.py",
            "native_progress.py",
            "protocol_freeze.py",
            "quality_cache.py",
            "quality_metrics.py",
            "quality_service.py",
            "run_modes.py",
            "validation/accuracy_gates.py",
            "workflow/execution_binding.py",
            "workflow/jobs.py",
            "workflow/deepx_build_binding.py",
            "workflow/scientific_reporting.py",
            "resources/remote_scripts/native_split_energy_preflight.py",
            "resources/remote_scripts/native_deepx_full_energy_hotloop.py",
            "resources/remote_scripts/native_producer_energy_plan.py",
        }.issubset(critical_modules),
        "critical_module_set_complete": build.get("critical_module_set_complete") is True,
        "package_content_digest": str(build.get("package_content_sha256") or "").startswith("sha256:"),
        "critical_code_digest": str(build.get("critical_code_digest_sha256") or "").startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v263-smoke",
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
