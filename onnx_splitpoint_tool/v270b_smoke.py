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
from .v270a_smoke import REQUIRED_FEATURES as V270A_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.runner import WORKFLOW_VERSION


REQUIRED_FEATURES = V270A_REQUIRED_FEATURES | {
    "hailo_metadata_via_hailo_python",
    "canonical_central_request_sha256",
    "role_scoped_tensorrt_quality_selector",
    "canonical_native_split_backend_receipt",
    "generic_full_model_identity_propagation",
    "smoke_energy_claim_source_clamp",
    "evaluated_matrix_claim_scope",
    "optional_ranking_generalization_scope",
    "scope_conditional_readiness",
    "canonical_confirmatory_holdout_runtime",
    "semantic_trt_cache_elapsed_exclusion",
    "yolov7_final_contract_canary",
}


def _hardware_smoke_repair_contract() -> bool:
    package_root = Path(__file__).resolve().parent
    runtime = (
        package_root / "runners" / "native_split_quality_runtime.py"
    ).read_text(encoding="utf-8")
    quality_chain = (package_root / "trt_quality_chain.py").read_text(
        encoding="utf-8"
    )
    split_quality = (package_root / "native_split_quality.py").read_text(
        encoding="utf-8"
    )
    suite = (
        package_root / "resources" / "templates" / "benchmark_suite.py.txt"
    ).read_text(encoding="utf-8")
    case_runner = (
        package_root / "resources" / "templates"
        / "run_split_onnxruntime.py.txt"
    ).read_text(encoding="utf-8")
    collector = (package_root / "energy" / "collector.py").read_text(
        encoding="utf-8"
    )
    energy_cli = (
        package_root / "resources" / "remote_scripts"
        / "energy_measurement_cli.py"
    ).read_text(encoding="utf-8")
    return all((
        'os.environ.get("HAILO_PY")' in runtime,
        "native_split_quality_hailort_probe_failed" in runtime,
        "def _sha256_token" in quality_chain,
        "declares_trt_full_quality_role" in quality_chain,
        "def canonical_native_split_backend" in split_quality,
        'cmd += ["--model-id"' in suite,
        'or full_path.stem' in case_runner,
        "def _clamp_diagnostic_claim_fields" in collector,
        'p.add_argument("--diagnostic-only"' in energy_cli,
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
        "hardware_smoke_repair_contract": _hardware_smoke_repair_contract(),
        "nonempty_critical_module_set": bool(
            build.get("critical_module_sha256")
        ),
    }
    result = {
        "schema": "onnx-splitpoint/v270b-smoke",
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
