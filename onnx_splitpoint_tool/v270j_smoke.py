"""Hardware-independent release-contract smoke test for version 2.70j."""
from __future__ import annotations

import ast
import json
import math
from pathlib import Path

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v269a_smoke import _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .v270i_smoke import REQUIRED_FEATURES as V270I_REQUIRED_FEATURES
from .native_split_quality import canonical_native_split_backend
from .validation.accuracy_gates import (
    AccuracyGatePolicy,
    apply_accuracy_gate_to_row,
    evaluate_detection_similarity,
)
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270I_REQUIRED_FEATURES | {
    "portable_native_split_binding_report_replay",
    "versioned_native_detection_similarity_policy",
    "orthogonal_structure_numeric_task_quality_axes",
    "lossless_completed_endpoint_reporting",
}


def _function_source(source: str, name: str) -> str:
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if node.name == name:
                return ast.get_source_segment(source, node) or ""
    return ""


def _p1_evidence_contract_checks() -> dict[str, bool]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    reporter = (
        project_root / "scripts" / "native_producer_final_report.py"
    ).read_text(encoding="utf-8")
    validator = (
        project_root / "scripts" / "native_producer_validate_visualize.py"
    ).read_text(encoding="utf-8")
    scientific = (
        package_root / "workflow" / "scientific_reporting.py"
    ).read_text(encoding="utf-8")
    concise = (
        package_root / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    performance = (
        package_root / "native_performance_reporting.py"
    ).read_text(encoding="utf-8")
    artifacts = (
        package_root / "workflow" / "artifacts.py"
    ).read_text(encoding="utf-8")
    probe = (
        project_root / "scripts"
        / "native_yolo_full_self_reference_probe.py"
    ).read_text(encoding="utf-8")
    updater = (
        project_root / "scripts" / "update_evalset_native_producers.py"
    ).read_text(encoding="utf-8")

    binding_function = _function_source(
        reporter, "_verify_split_semantic_artifacts"
    )
    claim_fields = _function_source(reporter, "_claim_contract_fields")
    concise_function = _function_source(
        concise, "_native_concise_summary_v60w"
    )
    performance_function = _function_source(
        performance, "collect_native_performance_matrix"
    )
    measurement_key = _function_source(
        performance, "_measurement_contract_key"
    )
    csv_function = _function_source(artifacts, "write_csv")

    policy = AccuracyGatePolicy()
    accepted = evaluate_detection_similarity({
        "ref_count": 11,
        "matched": 9,
        "mean_iou": 0.984,
        "iou_threshold": 0.50,
    }, {}, policy)
    rejected = evaluate_detection_similarity({
        "ref_count": 11,
        "matched": 8,
        "mean_iou": 0.984,
        "iou_threshold": 0.50,
    }, {}, policy)
    low_iou = evaluate_detection_similarity({
        "ref_count": 11,
        "matched": 9,
        "mean_iou": 0.849,
        "iou_threshold": 0.50,
    }, {}, policy)
    legacy = AccuracyGatePolicy.from_mapping({"schema_version": 2})
    legacy_result = evaluate_detection_similarity({
        "ref_count": 11,
        "matched": 9,
        "mean_iou": 0.984,
        "iou_threshold": 0.50,
    }, {}, legacy)
    separated = {
        "task": "detection",
        "buildable": True,
        "runtime_executable": True,
        "structural_contract_pass": True,
        "strict_boundary_numeric_pass": False,
        "task_quality_gate": {
            "decision": "pass",
            "tier": "final",
            "primary": {"metric": "coco_ap_50_95"},
        },
    }
    apply_accuracy_gate_to_row(separated, policy)

    return {
        "portable_split_binding_report_replay": all((
            canonical_native_split_backend(
                "hailo10_to_tensorrt"
            ) == "hailo10h_to_trt",
            canonical_native_split_backend(
                "deepx_m1_to_tensorrt"
            ) == "deepx_to_trt",
            "_canonical_native_split_backend(" in binding_function,
            "_identity_value('source_run_id')" in binding_function,
            "verification_mode='portable'" in binding_function,
            "native_split_final_portable_binding_valid"
            in binding_function,
        )),
        "versioned_detection_similarity_policy": all((
            policy.schema_version >= 3,
            policy.native_self_reference_policy_id
            == "class_aware_iou50_postnms_v2",
            math.isclose(
                policy.native_self_reference_min_match, 0.80
            ),
            math.isclose(
                policy.native_self_reference_min_mean_iou, 0.85
            ),
            accepted.get("numerical_similarity_pass") is True,
            rejected.get("numerical_similarity_pass") is False,
            low_iou.get("numerical_similarity_pass") is False,
            legacy.native_self_reference_min_match == 0.90,
            legacy_result.get("numerical_similarity_pass") is False,
            "evaluate_detection_similarity(" in validator,
            "evaluate_detection_similarity(" in probe,
            "--quality-gate-json" in updater,
        )),
        "orthogonal_evidence_axes": all((
            separated.get("contract_consistent") is True,
            separated.get("structural_contract_pass") is True,
            separated.get("numerical_similarity_pass") is False,
            separated.get("task_quality_pass") is True,
            set(separated.get("evidence_axes") or {}) == {
                "structural_contract",
                "numerical_similarity",
                "task_quality",
            },
            all(
                name in claim_fields
                for name in (
                    "structural_contract_pass",
                    "numerical_similarity_pass",
                    "task_quality_pass",
                )
            ),
            all(
                name in concise_function
                for name in (
                    "structural_contract_pass",
                    "numerical_similarity_pass",
                    "task_quality_pass",
                )
            ),
            all(
                name in performance_function
                for name in (
                    "structural_contract_pass",
                    "numerical_similarity_pass",
                    "task_quality_pass",
                )
            ),
        )),
        "lossless_endpoint_reporting": all((
            all(
                name in claim_fields
                for name in (
                    "source_e2e_scope",
                    "e2e_scope",
                    "completed_task_endpoint_attestation",
                    "frozen_host_postprocess_contract",
                )
            ),
            all(
                name in concise_function
                for name in (
                    "source_e2e_scope",
                    "e2e_scope",
                    "completed_task_endpoint_attestation",
                    "frozen_host_postprocess_contract",
                )
            ),
            all(
                name in measurement_key
                for name in (
                    "e2e_scope",
                    "completed_task_endpoint_contract_hash",
                    "frozen_host_postprocess_contract_sha256",
                )
            ),
            "native_performance_observations.json" in scientific,
            "native_e2e_scope_counts" in scientific,
            "native_completed_endpoint_attested_count" in scientific,
            "native_frozen_host_postprocess_count" in scientific,
            '("e2e_scope", "E2E scope")' in scientific,
            "completed_task_output_endpoint_id" in scientific,
            "Completed endpoint" in scientific,
            '("host_postprocess_frozen", "Host tail frozen")'
            in scientific,
            "json.dumps(" in csv_function,
            "Mapping, list, tuple" in csv_function,
        )),
    }


def main() -> int:
    build = package_build_snapshot()
    p1_contracts = _p1_evidence_contract_checks()
    checks = {
        "version": __version__ in {"2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "release": __release__ in {"2.75.40", "2.75.41", "2.75.42", "2.75.46", "2.75.47"},
        "lineage": __development_lineage__ in {"v2.75.40", "v2.75.41", "v2.75.42", "v2.75.46", "v2.75.47"},
        "workflow": (
            WORKFLOW_VERSION in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}
        ),
        "build_id": (
            __build_id__ in {"v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair", "v2.75.41-deepx-calibration-500-vs-1000-canary", "v2.75.42-real-evidence-and-quality-companion-repair", "v2.75.46-native-full-onnx-attestation-standard-quality-projection", "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"}
        ),
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "source_mirrors": (
            _source_mirrors_match()
            and _all_applicable_remote_source_mirrors_match()
        ),
        "critical_module_set_complete": (
            build.get("critical_module_set_complete") is True
        ),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
        "p1_evidence_contracts": all(p1_contracts.values()),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270j-smoke",
        "schema_version": 1,
        "ok": all(checks.values()),
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "checks": checks,
        "p1_contracts": p1_contracts,
        "build": build,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
