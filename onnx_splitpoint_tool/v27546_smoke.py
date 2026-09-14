"""Hardware-independent release contract for ONNX Split-Point Tool 2.75.46."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from . import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION
from .workflow.scientific_reporting import project_central_quality_status


VERSION = "2.75.46"
BUILD_ID = (
    "v2.75.46-native-full-onnx-attestation-standard-quality-projection"
)
CURRENT_VERSION = "2.75.47"
CURRENT_BUILD_ID = (
    "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
)
NEW_FEATURES = {
    "native_full_selected_onnx_interpreter_attestation",
    "native_full_onnx_attestation_exception_diagnostics",
    "standard_setup_local_quality_contract_projection",
    "generic_quality_diagnostic_row_separation",
    "deepx_calibration_size_output_contract_hash_normalization",
    "deepx_calibration_size_semantic_endpoint_invariant",
}
RETAINED_V27545_FEATURES = {
    "gui_large_audit_start_confirmation",
    "audit_minimum_valid_bound",
    "gui_explicit_deepx_classification_preprocessing",
    "large_audit_active_trt_working_set_admission",
    "retained_trt_cache_budget_separation",
    "large_audit_resume_cache_reuse",
}
REQUIRED_FEATURES = NEW_FEATURES | RETAINED_V27545_FEATURES
REQUIRED_FILES = (
    "profiles/resnet_yolo26s_yolov7_v27546_standard_anchor_b500.yaml",
    "onnx_splitpoint_tool/v27545_smoke.py",
    "onnx_splitpoint_tool/v27546_smoke.py",
    "scripts/run_v27545_small_acceptance.sh",
    "scripts/run_v27546_small_acceptance.sh",
    "tests/test_v27545_large_audit_working_set_admission.py",
    "tests/test_v27545_release_provenance.py",
    "tests/test_v27546_hailo_onnx_interpreter.py",
    "tests/test_v27546_calibration_size_canary_output_contract.py",
    "tests/test_v27546_standard_quality_projection.py",
    "tests/test_v27546_release_provenance.py",
    "TESTANLEITUNG_2.75.45.md",
    "VERSION_2.75.45_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.46.md",
    "VERSION_2.75.46_BUILD_AND_TEST_REPORT.md",
)
CRITICAL_MODULES = {
    "deepx/calibration_size_canary.py",
    "resources/remote_scripts/native_full_baseline_eval_runner.py",
    "workflow/scientific_reporting.py",
}
_STANDARD_SCHEMA = (
    "onnx-splitpoint/standard-setup-local-tensorrt-quality-"
    "acceptance-identity-contract"
)
_IDENTITY_KEY_FIELDS = [
    "model_id",
    "source_run_id",
    "setup_id",
    "backend",
    "variant",
    "execution_role",
    "performance_claims_emitted",
]
_MODELS = ("resnet50", "yolo26s", "yolov7_paper")
_SETUPS = (
    ("orin_nx_hailo8_01", "hailo8"),
    ("orin_nx_hailo10_01", "hailo10h"),
    ("orin_nx_deepx_m1_01", "deepx"),
)


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _quality_row(
    *, model: str, run_id: str, setup: str, backend: str,
    variant: str, role: str, claims: object,
) -> dict[str, Any]:
    identity = "|".join((model, run_id, setup, backend, variant, role))
    task = "classification" if model == "resnet50" else "detection"
    primary_metric = "top1_accuracy" if task == "classification" else (
        "coco_ap_50_95"
    )
    guardrail_metric = "top5_accuracy" if task == "classification" else (
        "ap50"
    )
    return {
        "schema": "onnx-splitpoint/management-paired-quality-result",
        "status": "completed",
        "technical_status": "completed",
        "scientific_status": "pass",
        "decision": "pass",
        "model_id": model,
        "task": task,
        "case_id": "full" if variant == "full" else "selected",
        "variant": variant,
        "run_id": run_id,
        "source_run_id": run_id,
        "backend": backend,
        "setup_id": setup,
        "source_setup_id": setup,
        "execution_role": role,
        "performance_claims_emitted": claims,
        "evaluation_fingerprint": _digest("evaluation:" + identity),
        "source_request_sha256": "sha256:" + _digest("request:" + identity),
        "producer_identity_sha256": _digest("producer:" + identity),
        "producer_binding_eligible": True,
        "runtime_precision_identity": "fp16",
        "n": 500,
        "primary": {
            "metric": primary_metric,
            "candidate": 0.8,
            "reference": 0.8,
            "delta": 0.0,
            "ci_low": -0.001,
            "ci_high": 0.001,
            "margin": 0.01,
            "decision": "pass",
            "bootstrap_repetitions_requested": 500,
            "bootstrap_repetitions": 500,
        },
        "guardrails": {
            guardrail_metric: {
                "metric": guardrail_metric,
                "decision": "pass",
            }
        },
    }


def _standard_quality_projection_contract() -> bool:
    results: list[dict[str, Any]] = []
    diagnostics = (
        ("deepx_m1_full", "orin_nx_deepx_m1_01", "deepx_m1", "full"),
        ("deepx_to_trt", "orin_nx_deepx_m1_01", "deepx_to_trt", "composed"),
        ("hailo10", "orin_nx_hailo10_01", "hailo10", "full"),
        ("hailo10h_to_trt", "orin_nx_hailo10_01", "hailo10h_to_trt", "composed"),
        ("hailo8", "orin_nx_hailo8_01", "hailo8", "full"),
        ("hailo8_to_trt", "orin_nx_hailo8_01", "hailo8_to_trt", "composed"),
        ("ort_tensorrt", "orin_nx_deepx_m1_01", "ort_tensorrt", "full"),
        ("ort_tensorrt", "orin_nx_deepx_m1_01", "ort_tensorrt", "composed"),
    )
    for model in _MODELS:
        for setup, _producer in _SETUPS:
            results.append(_quality_row(
                model=model,
                run_id="native_full_tensorrt",
                setup=setup,
                backend="tensorrt",
                variant="full",
                role="full_quality_only",
                claims=False,
            ))
        for run_id, setup, backend, variant in diagnostics:
            results.append(_quality_row(
                model=model,
                run_id=run_id,
                setup=setup,
                backend=backend,
                variant=variant,
                role="",
                claims=False if variant == "composed" else None,
            ))
    contract = {
        "schema": _STANDARD_SCHEMA,
        "schema_version": 1,
        "execution_scope": "standard_quality_setup_local_tensorrt",
        "identity_key_fields": list(_IDENTITY_KEY_FIELDS),
        "model_ids": list(_MODELS),
        "expected_identities": [
            {
                "id": f"tensorrt_at_{producer}_full",
                "source_run_id": "native_full_tensorrt",
                "setup_id": setup,
                "backend": "tensorrt",
                "variant": "full",
                "execution_role": "full_quality_only",
                "performance_claims_emitted": False,
            }
            for setup, producer in _SETUPS
        ],
    }
    projected = project_central_quality_status({
        "schema": "onnx-splitpoint/central-quality-summary",
        "status": "ok",
        "request_count": len(results),
        "completed_count": len(results),
        "technical_completed_count": len(results),
        "failed_count": 0,
        "merge": {"unmatched_result_count": 0},
        "quality_acceptance_identity_contract": contract,
        "results": results,
    })
    return bool(
        len(results) == 33
        and projected.get("technical_status") == "ok"
        and projected.get("result_count") == 33
        and projected.get("aggregate_expected_full_result_count") == 9
        and projected.get("aggregate_full_result_count") == 9
        and projected.get("aggregate_usable_full_result_count") == 9
        and projected.get("aggregate_excluded_diagnostic_full_result_count")
        == 12
        and projected.get("aggregate_identity_contract_complete") is True
        and projected.get("aggregate_identity_contract_issue_count") == 0
        and projected.get("aggregate_decision_counts") == {"pass": 9}
        and projected.get("quality_decision") == "pass"
        and projected.get("scientific_pass") is True
    )


def _native_full_onnx_attestation_contract(root: Path) -> bool:
    source = root / "scripts/native_full_baseline_eval_runner.py"
    packaged = (
        root
        / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_full_baseline_eval_runner.py"
    )
    if not source.is_file() or not packaged.is_file():
        return False
    text = source.read_text(encoding="utf-8")
    markers = (
        "onnx-splitpoint/hailo-source-raw-head-onnx-probe",
        "ONNX_SPLITPOINT_RAW_HEAD_PROBE_V2=",
        "load_model_from_string(source_bytes)",
        '"source_onnx_sha256": source_sha256',
        '"failure_phase": "import_onnx"',
        '"failure_phase": "load_onnx"',
        '"exception_type"',
        '"exception_detail"',
        "_select_engine_python",
        "onnx_python: str | None = None",
        "diagnostics_out: dict[str, Any] | None = None",
        '"hailo_hef_build_receipt_diagnostics"',
        "ns.engine_python_selected",
        "hailo_source_raw_head_onnx_identity_drift",
        "validate_parent_identity_post_probe",
    )
    return source.read_bytes() == packaged.read_bytes() and all(
        marker in text for marker in markers
    )


def _deepx_calibration_size_contract(root: Path) -> bool:
    module = root / "onnx_splitpoint_tool/deepx/calibration_size_canary.py"
    regression = (
        root / "tests/test_v27546_calibration_size_canary_output_contract.py"
    )
    if not module.is_file() or not regression.is_file():
        return False
    text = module.read_text(encoding="utf-8")
    return all(
        marker in text
        for marker in (
            "<calibration-arm-output-contract-sha256>",
            "quality_record_endpoint",
            "quality_record_endpoint_contract_sha256",
            "quality_contract_sha256",
            "endpoint_contract_hash",
        )
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    build = package_build_snapshot()
    critical = set(build.get("critical_module_sha256") or {})
    checks = {
        "version": __version__ in {VERSION, CURRENT_VERSION},
        "release": __release__ in {VERSION, CURRENT_VERSION},
        "lineage": __development_lineage__
        in {f"v{VERSION}", f"v{CURRENT_VERSION}"},
        "workflow": WORKFLOW_VERSION in {BUILD_ID, CURRENT_BUILD_ID},
        "build_id": __build_id__ in {BUILD_ID, CURRENT_BUILD_ID},
        "build_contract": int(__build_contract_version__) == 2,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "release_files": all((root / name).is_file() for name in REQUIRED_FILES),
        "entry_points": all(
            marker in pyproject
            for marker in (
                f'version = "{CURRENT_VERSION}"',
                (
                    "onnx-splitpoint-smoke-v27547 = "
                    '"onnx_splitpoint_tool.v27547_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-47 = "
                    '"onnx_splitpoint_tool.v27547_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v27546 = "
                    '"onnx_splitpoint_tool.v27546_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-46 = "
                    '"onnx_splitpoint_tool.v27546_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v27545 = "
                    '"onnx_splitpoint_tool.v27545_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-45 = "
                    '"onnx_splitpoint_tool.v27545_smoke:main"'
                ),
            )
        ),
        "native_full_onnx_attestation": (
            _native_full_onnx_attestation_contract(root)
        ),
        "standard_quality_projection_9_plus_24": (
            _standard_quality_projection_contract()
        ),
        "deepx_calibration_size_output_contract": (
            _deepx_calibration_size_contract(root)
        ),
        "critical_inventory_complete": (
            build.get("critical_module_set_complete") is True
        ),
        "claim_critical_repairs": CRITICAL_MODULES.issubset(critical),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v27546-smoke",
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
