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
    FrozenDecodedNmsPostprocessor,
    build_completed_detection_comparison_endpoint_contract,
    build_frozen_decoded_nms_normalization_contract,
    build_frozen_postprocess_contract,
    build_normalized_detection_endpoint_attestation,
    tensor_signature,
    verify_completed_detection_comparison_endpoint_contract,
)
from .runners.harness.yolo import YOLOV7_PAPER_ONNX_SHA256
from .v269a_smoke import _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .v270k_smoke import REQUIRED_FEATURES as V270K_REQUIRED_FEATURES
from .validation.accuracy_gates import (
    AccuracyGatePolicy,
    apply_accuracy_gate_to_row,
)
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270K_REQUIRED_FEATURES | {
    "canonical_yolo26_regcls_validation",
    "completed_detection_endpoint_v2",
    "native_quality_hotloop_energy_handoff",
    "lossless_scientific_native_evidence",
    "structural_claim_fail_closed",
    "evaluated_matrix_claim_scope",
    "optional_ranking_generalization_scope",
    "scope_conditional_readiness",
    "canonical_confirmatory_holdout_runtime",
    "semantic_trt_cache_elapsed_exclusion",
    "yolov7_final_contract_canary",
}


def _completed_endpoint_contracts() -> dict[str, bool]:
    raw_outputs = {
        "hailo_small": np.full(
            (1, 3, 80, 80, 85), -20.0, dtype=np.float32,
        ),
        "hailo_medium": np.full(
            (1, 3, 40, 40, 85), -20.0, dtype=np.float32,
        ),
        "hailo_large": np.full(
            (1, 3, 20, 20, 85), -20.0, dtype=np.float32,
        ),
    }
    frozen_tail = build_frozen_postprocess_contract(
        model_id="yolov7_paper",
        model_sha256=YOLOV7_PAPER_ONNX_SHA256,
        outputs=raw_outputs,
        input_hw=[640, 640],
        original_wh=[80, 60],
    )
    raw_v2 = build_completed_detection_comparison_endpoint_contract(
        frozen_tail,
    )

    direct_outputs = {
        "output0": np.asarray(
            [[[8.0, 12.0, 40.0, 52.0, 0.9, 0.0]]],
            dtype=np.float32,
        ),
    }
    physical_hash = "a" * 64
    physical_attestation = {
        "schema": (
            "onnx-splitpoint/runtime-output-endpoint-attestation"
        ),
        "schema_version": 3,
        "attested": True,
        "status": "passed",
        "endpoint": "decoded_nms",
        "stage": "decoded_nms",
        "values_decoded_xyxy_score_class": True,
        "declaration_attested": True,
        "endpoint_contract_hash": physical_hash,
        "tensor_signature": tensor_signature(direct_outputs),
        "declared_contract": {
            "model_id": "yolov7_paper",
            "source_coordinate_space": (
                "model_input_letterbox_xyxy_pixels"
            ),
        },
    }
    direct_normalizer = (
        build_frozen_decoded_nms_normalization_contract(
            model_id="yolov7_paper",
            outputs=direct_outputs,
            input_hw=[640, 640],
            original_wh=[80, 60],
            preprocess={
                "mode": "letterbox",
                "color_space": "RGB",
                "pad_value": 114,
            },
            source_coordinate_space=(
                "model_input_letterbox_xyxy_pixels"
            ),
            source_endpoint_contract_hash=physical_hash,
            source_output_endpoint_attestation=physical_attestation,
        )
    )
    direct_v2 = build_completed_detection_comparison_endpoint_contract(
        direct_normalization_contract=direct_normalizer,
    )
    processor = FrozenDecodedNmsPostprocessor(direct_normalizer)
    result = processor.process(direct_outputs)
    completion = build_normalized_detection_endpoint_attestation(
        direct_normalizer,
        result,
        completed_frames=1,
        postprocess_completed_frames=1,
    )
    archived_v1 = (
        build_completed_detection_comparison_endpoint_contract(
            frozen_tail,
            schema_version=1,
        )
    )
    return {
        "v2_raw_and_direct_match": (
            raw_v2["output_endpoint_id"]
            == direct_v2["output_endpoint_id"]
        ),
        "v2_raw_verifies": (
            verify_completed_detection_comparison_endpoint_contract(
                raw_v2,
                frozen_contract=frozen_tail,
            )["schema_version"]
            == 2
        ),
        "v2_direct_verifies": (
            verify_completed_detection_comparison_endpoint_contract(
                direct_v2,
                direct_normalization_contract=direct_normalizer,
            )["schema_version"]
            == 2
        ),
        "v1_remains_verifiable": (
            verify_completed_detection_comparison_endpoint_contract(
                archived_v1,
                frozen_contract=frozen_tail,
            )["schema_version"]
            == 1
        ),
        "direct_completion_exact": (
            processor.completed_count == 1
            and completion["attested"] is True
            and completion["postprocess_completed_frames"] == 1
            and completion["completed_task_completion_mode"]
            == "integrated_accelerator_plus_frozen_normalization"
        ),
    }


def _claim_contract() -> bool:
    row = {
        "task": "detection",
        "buildable": True,
        "runtime_executable": True,
        "claim_ok": True,
        "status": "claim_ok",
        "interface_contract_pass": False,
        "strict_boundary_numeric_pass": True,
        "task_quality_gate": {
            "decision": "pass",
            "tier": "final",
            "primary": {"metric": "coco_ap_50_95"},
        },
    }
    apply_accuracy_gate_to_row(row, AccuracyGatePolicy())
    return bool(
        row.get("claim_ok") is False
        and row.get("structural_contract_pass") is False
        and row.get("numerical_similarity_pass") is True
        and row.get("task_quality_pass") is True
    )


def _source_contract_markers() -> dict[str, bool]:
    project_root = Path(__file__).resolve().parent.parent
    validator = (
        project_root / "scripts" / "native_producer_validate_visualize.py"
    ).read_text(encoding="utf-8")
    reporter = (
        project_root / "scripts" / "native_producer_final_report.py"
    ).read_text(encoding="utf-8")
    energy = (
        project_root / "scripts" / "native_producer_energy_plan.py"
    ).read_text(encoding="utf-8")
    concise = (
        project_root / "onnx_splitpoint_tool" / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    scientific = (
        project_root
        / "onnx_splitpoint_tool"
        / "workflow"
        / "scientific_reporting.py"
    ).read_text(encoding="utf-8")
    return {
        "yolo26_regcls_decoder": all(
            marker in validator
            for marker in (
                "ultralytics_regcls",
                "canonical_multiscale:raw",
                "decoder_format",
                "max_det=300",
            )
        ),
        "quality_and_hotloop_handoff": all(
            marker in reporter + energy
            for marker in (
                "_SUPPORTED_QUALITY_SUMMARY_SCHEMA_VERSIONS",
                "tensorrt_full_completed_task_hotloop",
                "frozen_decoded_nms_normalization_contract",
            )
        ),
        "lossless_scientific_fields": all(
            marker in concise + scientific
            for marker in (
                "numerical_similarity_mean_iou",
                "numerical_similarity_mean_iou_threshold",
                "e2e_claim_eligible",
                "host_postprocessing_evidence_source",
                "completed_task_comparison_endpoint_contract",
            )
        ),
    }


def main() -> int:
    build = package_build_snapshot()
    endpoint_contracts = _completed_endpoint_contracts()
    source_contracts = _source_contract_markers()
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
        "features": REQUIRED_FEATURES.issubset(
            set(__build_features__)
        ),
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
        "endpoint_contracts": all(endpoint_contracts.values()),
        "source_contracts": all(source_contracts.values()),
        "structural_claim_fail_closed": _claim_contract(),
    }
    result = {
        "schema": "onnx-splitpoint/v270l-smoke",
        "schema_version": 1,
        "ok": all(checks.values()),
        "passed": sum(bool(value) for value in checks.values()),
        "failed": sum(not bool(value) for value in checks.values()),
        "checks": checks,
        "endpoint_contracts": endpoint_contracts,
        "source_contracts": source_contracts,
        "build": build,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
