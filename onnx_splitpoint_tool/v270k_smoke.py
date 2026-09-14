"""Hardware-independent release-contract smoke test for version 2.70l."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .native_detection_postprocess import (
    build_completed_detection_comparison_endpoint_contract,
    build_frozen_postprocess_contract,
    verify_completed_detection_comparison_endpoint_contract,
)
from .runners.harness.yolo import YOLOV7_PAPER_ONNX_SHA256
from .v269a_smoke import _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .v270j_smoke import REQUIRED_FEATURES as V270J_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270J_REQUIRED_FEATURES | {
    "sealed_rank3_full_self_reference_feed",
    "native_single_input_reject_backfill",
    "canonical_host_postprocessing_alias",
    "backend_independent_completed_task_endpoint",
}


def _comparison_endpoint_id(prefix: str) -> str:
    outputs = {
        f"{prefix}_small": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32
        ),
        f"{prefix}_medium": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32
        ),
        f"{prefix}_large": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32
        ),
    }
    frozen = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
    )
    comparison = build_completed_detection_comparison_endpoint_contract(
        frozen
    )
    verified = verify_completed_detection_comparison_endpoint_contract(
        comparison,
        frozen_contract=frozen,
    )
    return str(verified["output_endpoint_id"])


def _native_contract_completion_checks() -> dict[str, bool]:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    validator = (
        project_root / "scripts" / "native_producer_validate_visualize.py"
    ).read_text(encoding="utf-8")
    execution_plan = (
        package_root / "execution_plan.py"
    ).read_text(encoding="utf-8")
    runner = (
        package_root / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    host_normalizer = (
        package_root / "validation" / "host_postprocess.py"
    ).read_text(encoding="utf-8")
    gate_sources = [
        (package_root / "validation" / "accuracy_gates.py").read_text(
            encoding="utf-8"
        ),
        (package_root / "validation" / "gates.py").read_text(
            encoding="utf-8"
        ),
        (package_root / "benchmark" / "accuracy_gate.py").read_text(
            encoding="utf-8"
        ),
        (package_root / "benchmark" / "accuracy_gates.py").read_text(
            encoding="utf-8"
        ),
    ]
    reporter = (
        project_root / "scripts" / "native_producer_final_report.py"
    ).read_text(encoding="utf-8")
    performance = (
        package_root / "native_performance_reporting.py"
    ).read_text(encoding="utf-8")
    scientific = (
        package_root / "workflow" / "scientific_reporting.py"
    ).read_text(encoding="utf-8")

    endpoint_a = _comparison_endpoint_id("hailo8")
    endpoint_b = _comparison_endpoint_id("deepx")

    return {
        "sealed_rank3_full_self_reference_feed": all(
            marker in validator
            for marker in (
                "sealed_runtime_to_onnx_reference_v1",
                "runtime_input_sha256",
                "source_layout",
                "target_layout",
                "derived_reference_tensor_sha256",
                "sealed_runtime_identity_dtype_mismatch",
            )
        ),
        "native_single_input_reject_backfill": all(
            (
                "reject_and_backfill_from_frozen_prediction"
                in execution_plan,
                "effective_require_single_part2_input"
                in execution_plan,
                "native_backfill_replaces_case" in runner,
                "native_backfill_scope" in runner,
                "native_capability_backfill_count" in runner,
            )
        ),
        "canonical_host_postprocessing_alias": all(
            (
                "resolve_host_postprocess_evidence" in host_normalizer,
                "apply_host_postprocess_aliases" in host_normalizer,
                "host_postprocess_evidence_conflict" in host_normalizer,
                "host_postprocessing_available" in host_normalizer,
                "host_tail_available" in host_normalizer,
                all(
                    "resolve_host_postprocess_evidence" in source
                    for source in gate_sources
                ),
            )
        ),
        "backend_independent_completed_task_endpoint": all(
            (
                endpoint_a == endpoint_b,
                endpoint_a.startswith(
                    "detection:decoded_nms:comparison:"
                ),
                "completed_task_comparison_output_endpoint_id"
                in reporter,
                "comparison_output_endpoint_id" in performance,
                "comparison_output_endpoint_id" in scientific,
                "physical_output_endpoint_id" in scientific,
            )
        ),
    }


def main() -> int:
    build = package_build_snapshot()
    repair_contracts = _native_contract_completion_checks()
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
        "native_contract_completion_repairs": all(
            repair_contracts.values()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270k-smoke",
        "schema_version": 1,
        "ok": all(checks.values()),
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "checks": checks,
        "repair_contracts": repair_contracts,
        "build": build,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
