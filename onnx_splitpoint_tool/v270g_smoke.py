"""Hardware-independent release-contract smoke test for version 2.70g."""
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
from .v270f_smoke import REQUIRED_FEATURES as V270F_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270F_REQUIRED_FEATURES | {
    "quality_first_generic_bridge_memory_order",
    "portable_native_trt_metadata_validation",
    "exact_native_probe_artifact_identity",
    "canonical_yolov7_multiscale_native_validation",
    "model_scoped_detection_reference_images",
}


def _native_validation_bridge_repairs_are_bound() -> bool:
    package_root = Path(__file__).resolve().parent
    project_root = package_root.parent
    runtime = (
        package_root / "runners" / "native_split_quality_runtime.py"
    ).read_text(encoding="utf-8")
    template = (
        package_root / "resources" / "templates"
        / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    validator = (
        project_root / "scripts" / "native_producer_validate_visualize.py"
    ).read_text(encoding="utf-8")
    probe = (
        project_root / "scripts"
        / "native_yolo_full_self_reference_probe.py"
    ).read_text(encoding="utf-8")
    console = (
        project_root / "scripts" / "native_console_smoke.py"
    ).read_text(encoding="utf-8")
    return all((
        "def prepare_quality_first_boundary_input(" in runtime,
        "tuple(memory_shape[index] for index in perm) != target" in runtime,
        "quality_first_internal_bridge_memory_order_preserved" in template,
        "native split Quality-FIRST runtime input name mismatch" in template,
        "portable_native_split_quality_binding" in validator,
        "def _cached_probe_matches_native_output(" in validator,
        "'kind': 'raw_yolo_multiscale'" in validator,
        "external_reference_input_mismatch" in validator,
        "onnx-splitpoint/native-yolo-full-self-reference" in probe,
        '"native_output_manifest_sha256"' in probe,
        "tests/test_v270g_smoke_followup_repairs.py" in console,
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
        "native_validation_bridge_repairs": (
            _native_validation_bridge_repairs_are_bound()
        ),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270g-smoke",
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
