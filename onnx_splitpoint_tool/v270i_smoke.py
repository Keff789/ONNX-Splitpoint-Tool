"""Hardware-independent release-contract smoke test for version 2.70i."""
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
from .v269a_smoke import _source_mirrors_match
from .v269b_smoke import _all_applicable_remote_source_mirrors_match
from .v270h_smoke import REQUIRED_FEATURES as V270H_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270H_REQUIRED_FEATURES | {
    "native_full_input_manifest_self_reference",
    "strict_native_full_input_evidence_binding",
    "independent_native_execution_semantic_status",
    "completed_task_endpoint_timing_parity",
    "authoritative_yolov7_full_decoder_identity",
    "fail_closed_native_evidence_status_axes",
}


def _native_full_evidence_repairs_are_bound() -> bool:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    validator = (
        project_root / "scripts" / "native_producer_validate_visualize.py"
    ).read_text(encoding="utf-8")
    full_runner = (
        project_root / "scripts" / "native_full_baseline_eval_runner.py"
    ).read_text(encoding="utf-8")
    semantic_dump = (
        project_root / "scripts" / "native_full_semantic_dump.py"
    ).read_text(encoding="utf-8")
    trt_hotloop = (
        project_root / "scripts" / "native_trt_full_completed_hotloop.py"
    ).read_text(encoding="utf-8")
    suite_template = (
        package_root / "resources" / "templates" / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    evidence_status = (
        package_root / "workflow" / "evidence_status.py"
    ).read_text(encoding="utf-8")
    runner = (
        package_root / "workflow" / "runner.py"
    ).read_text(encoding="utf-8")
    return all((
        "def _find_native_full_input_manifest_for_output(" in validator,
        "runtime_input_sha256" in validator,
        "native_full_input_manifest_missing_or_invalid" in validator,
        '"input_manifest_sha256": _sha256(input_manifest)' in semantic_dump,
        "def _attach_trt_completed_task_hotloop(" in full_runner,
        '"kind": "tensorrt_full_completed_task_hotloop"' in full_runner,
        '"completed_task_stage": "decoded_nms"' in full_runner,
        '"measurement_concurrency": 1' in full_runner,
        "tensorrt_full_completed_task_hotloop" in trt_hotloop,
        "def _deepx_contract_model_id(" in suite_template,
        "native_full_raw_detection_model_identity_conflict" in suite_template,
        "def derive_native_evidence_status(" in evidence_status,
        "def blocking_status(" in evidence_status,
        'ok=(status == "ok")' in runner,
        'reports / "native_evidence_status.json"' in runner,
    ))


def main() -> int:
    build = package_build_snapshot()
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
        "native_full_evidence_repairs": (
            _native_full_evidence_repairs_are_bound()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270i-smoke",
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
