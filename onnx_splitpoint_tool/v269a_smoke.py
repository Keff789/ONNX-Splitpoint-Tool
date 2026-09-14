"""2.69a compatibility smoke evaluated against the current Tool build."""
from __future__ import annotations

import json
from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .run_modes import apply_run_mode, default_run_modes_config
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = {
    "native_level_playing_field_timing",
    "native_median_ci_repetitions",
    "native_host_power_clock_thermal_evidence",
    "native_endpoint_precision_claim_gates",
    "exact_trtexec_work_contract",
    "common_pinned_trt_part2_consumer",
    "observed_native_completion_work_units",
    "owned_deepx_boundary_payload",
    "canonical_native_endpoint_contract_hash",
    "fail_closed_cross_runner_claim_identity",
    "raw_repetition_record_claim_gate",
    "capture_group_bound_host_telemetry",
    "verified_remote_runner_dependency_closure",
    "raw_multiscale_cpu_reference_adapter",
    "diagnostic_independent_quality_result_join",
    "large_canonical_csv_audit",
    "deepx_full_central_quality_request",
    "central_quality_status_breakdown",
}


MIRRORED_SCRIPTS = (
    "native_deepx_trt_e2e_from_benchmarkset.py",
    "native_fifo_eval_runner.py",
    "native_fifo_smoke_matrix.py",
    "native_full_baseline_eval_runner.py",
    "native_full_semantic_dump.py",
    "native_hailo10_trt_e2e_from_benchmarkset.py",
    "native_hailo_trt_fifo_from_benchmarkset.py",
    "native_host_telemetry.py",
    "native_producer_e2e_eval_runner.py",
    "native_producer_final_report.py",
    "native_producer_validate_visualize.py",
    "native_trt_from_benchmarkset.py",
    "run_evalrun_native_producer_variants.py",
    "smoke_hailo10_full_from_benchmarkset.py",
    "smoke_hailo10_hef_runner.py",
    "update_evalset_native_producers.py",
)


def _source_mirrors_match() -> bool:
    root = Path(__file__).resolve().parents[1]
    return all(
        (root / "scripts" / name).is_file()
        and (root / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name).is_file()
        and (root / "scripts" / name).read_bytes()
        == (root / "onnx_splitpoint_tool" / "resources" / "remote_scripts" / name).read_bytes()
        for name in MIRRORED_SCRIPTS
    )


def _performance_repetition_defaults_are_exact() -> bool:
    expected = {"smoke": 1, "standard": 3, "final": 3}
    config = default_run_modes_config()
    for mode, repetitions in expected.items():
        resolved, _audit = apply_run_mode(
            {"name": "v269_smoke", "execution_preset": {"id": mode}},
            mode_id=mode,
            config=config,
        )
        native = resolved.get("native_producers") or {}
        if int(native.get("repetitions") or 0) != repetitions:
            return False
    return True


def main() -> int:
    build = package_build_snapshot()
    checks = {
        "version": __version__ in {"2.69.6", "2.70.6", "2.70.7", "2.70.8", "2.70.9", "2.70.10", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "release": __release__ in {"2.69f", "2.70f", "2.70g", "2.70h", "2.70i", "2.70j", "2.71.1", "2.71.2", "2.71.3", "2.71.4", "2.72.0", "2.72.1", "2.72.2", "2.72.3", "2.72.4", "2.72.5", "2.72.6", "2.72.7", "2.73.0", "2.73.6", "2.73.7", "2.75.2", "2.75.3", "2.75.7", "2.75.8", "2.75.9", "2.75.10", "2.75.11", "2.75.12", "2.75.13", "2.75.14", "2.75.15", "2.75.16", "2.75.17", "2.75.20", "2.75.21", "2.75.22", "2.75.24", "2.75.25", "2.75.26", "2.75.27", "2.75.28", "2.75.30", "2.75.31", "2.75.32", "2.75.38", "2.75.39", "2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "lineage": __development_lineage__ in {"v2.69f", "v2.70f", "v2.70g", "v2.70h", "v2.70i", "v2.70j", "v2.71.1", "v2.71.2", "v2.71.3", "v2.71.4", "v2.72.0", "v2.72.1", "v2.72.2", "v2.72.3", "v2.72.4", "v2.72.5", "v2.72.6", "v2.72.7", "v2.73.0", "v2.73.6", "v2.73.7", "v2.75.2", "v2.75.3", "v2.75.7", "v2.75.8", "v2.75.9", "v2.75.10", "v2.75.11", "v2.75.12", "v2.75.14", "v2.75.16", "v2.75.17", "v2.75.20", "v2.75.21", "v2.75.22", "v2.75.24", "v2.75.25", "v2.75.26", "v2.75.27", "v2.75.28", "v2.75.30", "v2.75.31", "v2.75.32", "v2.75.38", "v2.75.39", "v2.75.40", "v2.75.41", "v2.75.42", "v2.75.46", "v2.75.47"},
        "workflow": WORKFLOW_VERSION in {"v2.69f-hardware-smoke-native-energy-repair", "v2.70f-legacy-log-callback-repair", "v2.70g-native-validation-bridge-repair", "v2.70h-remote-suite-bootstrap-repair", "v2.70i-native-full-evidence-repair", "v2.70j-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"},
        "build_id": __build_id__ in {"v2.69.6-hardware-smoke-native-energy-repair", "v2.70.6-legacy-log-callback-repair", "v2.70.7-native-validation-bridge-repair", "v2.70.8-remote-suite-bootstrap-repair", "v2.70.9-native-full-evidence-repair", "v2.70.10-native-evidence-reporting-repair", "v2.71.1-live-evidence-binding-repair", "v2.71.2-evidence-status-energy-resume-repair", "v2.71.3-resume-artifact-restage-repair", "v2.71.4-resume-cleanup-root-rehydration", "v2.72.0-completed-detection-evidence-contracts", "v2.72.1-mixed-runtime-energy-contract-repair", "v2.72.2-offline-energy-completed-consumer-repair", "v2.72.3-campaign-gate-quality-binding-repair", "v2.72.4-campaign-gate-quality-contract-mirror-repair", "v2.72.5-campaign-gate-hailo8-resume-contract-repair", "v2.72.6-campaign-gate-hailo8-preflight-artifact-rehydration-repair", "v2.72.7-native-energy-runtime-success-admission-repair", "v2.73.0-completed-quality-evidence-repair", "v2.73.6-single-writer-process-tree-cancellation", "v2.73.7-atomic-stage-energy-checkpoints", "v2.75.2-native-runtime-closure-partial-energy-repair", "v2.75.3-cache-verify-only-gate", "v2.75.7-cache-canary-remote-runtime-closure-repair", "v2.75.8-native-runtime-holdout-contract-repair", "v2.75.9-smoke-cache-reuse-repair", "v2.75.10-historical-cache-multihit-replay-repair", "v2.75.11-canonical-native-result-verification-repair", "v2.75.12-canonical-native-result-status-alias-repair", "v2.75.13-native-full-contract-line-repair", "v2.75.14-vendor-full-nine-row-repair", "v2.75.15-vendor-full-field-contract-repair", "v2.75.16-read-only-run-storage-admission", "v2.75.17-vendor-quality-diagnostic-hailo-alias-repair", "v2.75.20-remote-boundary-plan-finalization-repair", "v2.75.21-setup-local-tensorrt-quality-dispatch-repair", "v2.75.22-quality-replay-reporting-repair", "v2.75.24-standard-hef-preupload-gate", "v2.75.26-trt-quality-join-mean-iou-repair", "v2.75.27-final-claim-scope-cache-key-repair", "v2.75.28-final-campaign-bootstrap-energy-provenance-repair", "v2.75.30-score-independent-native-ranking-audit", "v2.75.31-compact-debug-ranking-gui-repair", "v2.75.32-audit-ranking-e2e-repair", "v2.75.38-deepx-full-quality-dxnn-dispatch-repair", "v2.75.39-pre-mutation-remote-failure-quarantine-repair", "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"},
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "performance_repetition_defaults": _performance_repetition_defaults_are_exact(),
        "source_mirrors": _source_mirrors_match(),
        "critical_module_set_complete": build.get("critical_module_set_complete") is True,
        "package_content_digest": str(build.get("package_content_sha256") or "").startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v269a-smoke",
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
