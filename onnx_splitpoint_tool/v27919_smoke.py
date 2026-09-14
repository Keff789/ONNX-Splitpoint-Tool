"""Hardware-independent smoke for the focused v2.79.19 maintenance release."""
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
from .release_identity import BUILD_ID, DEVELOPMENT_LINEAGE, VERSION
from .workflow.runner import WORKFLOW_VERSION


LINEAGE = DEVELOPMENT_LINEAGE
NEW_FEATURES = {
    "v27919_calibration_plausibility_warnings",
    "v27919_evalrun_remote_runtime_closure",
    "v27919_quality_request_contract_propagation",
    "v27919_accepted_case_required_scope",
    "v27919_artifact_index_report_reconciliation",
    "v27919_classification_native_split_policy",
    "v27919_classification_native_split_orchestration",
    "v27919_hailo8_float32_vstream_quantized_hef_storage",
    "v27919_post_dispatch_result_retention",
    "v27919_deepx_input_contract_model_binding",
    "v27919_deepx_result_binding_reason_codes",
}
REQUIRED_FEATURES = set(NEW_FEATURES)


def _read(root: Path, relative: str) -> str:
    return (root / relative).read_text(encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    platform_source = _read(root, "onnx_splitpoint_tool/platform_power.py")
    panel_source = _read(
        root, "onnx_splitpoint_tool/gui/panels/panel_hardware.py"
    )
    closure_source = _read(
        root, "onnx_splitpoint_tool/remote_runtime_closure.py"
    )
    template_source = _read(
        root,
        "onnx_splitpoint_tool/resources/templates/run_split_onnxruntime.py.txt",
    )
    runner_source = _read(root, "onnx_splitpoint_tool/workflow/runner.py")
    native_policy_source = _read(
        root, "onnx_splitpoint_tool/native_split_quality.py"
    )
    native_runtime_source = _read(
        root,
        "onnx_splitpoint_tool/runners/native_split_quality_runtime.py",
    )
    execution_source = _read(
        root, "onnx_splitpoint_tool/workflow/execution_binding.py"
    )
    deepx_build_source = _read(
        root, "onnx_splitpoint_tool/workflow/deepx_build_binding.py"
    )
    pyproject = _read(root, "pyproject.toml")
    generic_smoke = _read(root, "onnx_splitpoint_tool/v279_smoke.py")
    generic_acceptance = _read(root, "scripts/run_v279_small_acceptance.sh")
    local_acceptance = _read(root, "scripts/run_local_acceptance.sh")

    quality_fields = (
        "decoder_contract_sha256",
        "nms_contract_sha256",
        "quality_record_endpoint_contract_sha256",
    )
    checks = {
        "version": __version__ == __release__ == VERSION == "2.79.19",
        "lineage": __development_lineage__ == LINEAGE == "v2.79",
        "build": (
            __build_id__
            == WORKFLOW_VERSION
            == BUILD_ID
            == "v2.79.19-calibration-warning-evalrun-closure"
        ),
        "focused_features": REQUIRED_FEATURES.issubset(
            set(__build_features__)
        ),
        "calibration_warning_contract": all(
            marker in platform_source
            for marker in (
                '"warning_reasons": warning_reasons',
                '"pass": not reasons',
                "point_factor_spread_too_high",
                "scale_factor_outside_configured_range",
                "reference_current_outside_target_tolerance",
            )
        ),
        "calibration_gui_warning_contract": all(
            marker in panel_source
            for marker in (
                "Technical validity",
                "Plausibility warnings (saving remains allowed)",
                "Save blocked: measurement technically invalid",
            )
        ),
        "remote_runtime_closure": all(
            marker in closure_source
            for marker in (
                "onnx_splitpoint_tool/hailo_attempt_receipts.py",
                "onnx_splitpoint_tool/hailo_timeout_policy.py",
            )
        ),
        "quality_request_duplicates": all(
            template_source.count(f'"{field}"') >= 2
            for field in quality_fields
        ),
        "accepted_case_scope": all(
            marker in runner_source
            for marker in (
                "_accepted_case_contract_v27919",
                "_seal_model_scope_from_accepted_cases_v27919",
                "final_benchmark_set_accepted_cases",
            )
        ),
        "report_index_reconciliation": all(
            marker in runner_source
            for marker in (
                "_validated_canonical_report_artifacts",
                "_reconcile_canonical_report_cleanup",
            )
        ),
        "classification_native_split_policy": all(
            marker in native_policy_source
            for marker in (
                'model.startswith("mobilenet")',
                'model.startswith("regnet")',
                'canonical_backend == "hailo8_to_trt" and task == "classification"',
            )
        ),
        "classification_native_split_orchestration": all(
            marker in runner_source
            for marker in (
                "_native_model_family_task_v27919",
                'for family in ("resnet", "mobilenet", "regnet")',
                "row.get(\"model\")\n                    )[1]",
            )
        ),
        "hailo8_float32_vstream_quantized_storage": all(
            marker in native_runtime_source
            for marker in (
                "_hailo_native_output_format_type_name",
                "native_format_type != expected_dtype",
                'tensor["hef_native_storage_dtype"]',
            )
        ),
        "post_dispatch_result_retention": all(
            marker in execution_source
            for marker in (
                "post_dispatch_processing_failure",
                "remote_result_processing_failed",
                "collected work was retained",
            )
        ),
        "deepx_input_contract_model_binding": (
            '"model_id": str(model_id)' in deepx_build_source
            and 'contract_family = "classification_logits"'
            in deepx_build_source
        ),
        "deepx_result_binding_reason_codes": all(
            marker in execution_source
            for marker in (
                "binding_failure_reasons",
                "input_contract_model_id_mismatch",
                "runtime_file_sha256_mismatch",
                "source_image_sha256_mismatch",
            )
        ),
        "entrypoints": (
            'version = "2.79.19"' in pyproject
            and (
                'onnx-splitpoint-smoke-v27919 = '
                '"onnx_splitpoint_tool.v27919_smoke:main"'
            ) in pyproject
            and (
                'onnx-splitpoint-smoke-v2-79-19 = '
                '"onnx_splitpoint_tool.v27919_smoke:main"'
            ) in pyproject
        ),
        "current_aliases": (
            "from .v27919_smoke import" in generic_smoke
            and "run_v27919_small_acceptance.sh" in generic_acceptance
            and "run_v27919_small_acceptance.sh" in local_acceptance
        ),
    }
    failed = [name for name, passed in checks.items() if not passed]
    payload = {
        "status": "FAIL" if failed else "PASS",
        "version": VERSION,
        "build_id": BUILD_ID,
        "failed": failed,
        "checks": checks,
    }
    print(json.dumps(payload, indent=2, sort_keys=True))
    if failed:
        return 1
    print("PASS v2.79.19 smoke")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
