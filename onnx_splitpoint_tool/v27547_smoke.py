"""Hardware-independent release contract for ONNX Split-Point Tool 2.75.47."""
from __future__ import annotations

import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any

import yaml

from . import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v27546_smoke import REQUIRED_FEATURES as V27546_REQUIRED_FEATURES
from .workflow.artifacts import package_build_snapshot
from .workflow.debug_pack_policy import discover_ranking_audit_evidence
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.75.47"
BUILD_ID = "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
NEW_FEATURES = {
    "model_bound_yolov7_anchor_contract",
    "official_coco_yolov7_decoder_ab_probe",
    "generic_native_yolov7_decoder_parity",
    "debug_pack_ranking_audit_intent_fix",
    "debug_pack_non_audit_completeness",
    "early_declared_model_sha256_admission",
}
RETAINED_V27546_FEATURES = set(V27546_REQUIRED_FEATURES)
REQUIRED_FEATURES = NEW_FEATURES | RETAINED_V27546_FEATURES
REQUIRED_FILES = (
    "profiles/yolov7_paper_v27547_standard_anchor_b500.yaml",
    "onnx_splitpoint_tool/v27546_smoke.py",
    "onnx_splitpoint_tool/v27547_smoke.py",
    "scripts/probe_yolov7_decoder_ab.py",
    "scripts/run_yolov7_decoder_ab_probe.sh",
    "scripts/native_producer_validate_visualize.py",
    (
        "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_producer_validate_visualize.py"
    ),
    "scripts/run_v27546_small_acceptance.sh",
    "scripts/run_v27547_small_acceptance.sh",
    "tests/test_v27546_release_provenance.py",
    "tests/test_v27547_yolov7_decoder_contract.py",
    "tests/test_v27547_model_hash_gate.py",
    "tests/test_v27547_release_integration.py",
    "tests/test_v27547_release_provenance.py",
    "TESTANLEITUNG_2.75.46.md",
    "VERSION_2.75.46_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.47.md",
    "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md",
)
CRITICAL_MODULES = {
    "campaign.py",
    "execution_plan.py",
    "native_detection_postprocess.py",
    "runners/harness/yolo.py",
    "resources/remote_scripts/native_producer_validate_visualize.py",
    "validation/official_coco.py",
    "workflow/debug_pack.py",
    "workflow/debug_pack_policy.py",
    "workflow/runner.py",
}
MODEL_SHA256 = (
    "7a13e66f91047cce0e251c05f64159646847e842af31d60441c63dcdfad7825d"
)
VALIDATOR_MIRROR_SHA256 = (
    "1e2e4f293a381338549a57da5ceab28c136fc330ef42ba4bec2632abf248080e"
)


def _yolov7_contract_sources(root: Path) -> bool:
    yolo = root / "onnx_splitpoint_tool/runners/harness/yolo.py"
    probe = root / "scripts/probe_yolov7_decoder_ab.py"
    wrapper = root / "scripts/run_yolov7_decoder_ab_probe.sh"
    validator = root / "scripts/native_producer_validate_visualize.py"
    validator_mirror = (
        root
        / "onnx_splitpoint_tool/resources/remote_scripts/"
        "native_producer_validate_visualize.py"
    )
    regression = root / "tests/test_v27547_yolov7_decoder_contract.py"
    if not all(
        path.is_file()
        for path in (
            yolo, probe, wrapper, validator, validator_mirror, regression,
        )
    ):
        return False
    yolo_text = yolo.read_text(encoding="utf-8")
    probe_text = probe.read_text(encoding="utf-8")
    wrapper_text = wrapper.read_text(encoding="utf-8")
    validator_bytes = validator.read_bytes()
    validator_text = validator_bytes.decode("utf-8")
    validator_sha256 = hashlib.sha256(validator_bytes).hexdigest()
    return all(
        marker in yolo_text
        for marker in (
            "build_yolov7_decoder_contract",
            "verify_yolov7_decoder_contract",
            "registered_yolov7_decoder_contract",
            "onnx-splitpoint/yolov7-model-bound-decoder",
            "yolov7_paper_standard_anchors_640_v1",
            "yolov7_paper_legacy_tiny_anchors_640_v1",
        )
    ) and all(
        marker in probe_text
        for marker in (
            "onnx-splitpoint/yolov7-decoder-ab-probe",
            "pycocotools",
            "--preflight-only",
            "official_coco",
            "EXPECTED_SELECTION_REQUEST_SHA256",
            "OFFICIAL_COCO_MAX_DETS = (1, 10, 100)",
            "official_coco_all_policies_completed",
            "paired_contract_diff_exactly_pre_registered",
            "standard_production_ap_50_95_at_least_020",
            "standard_production_ap50_at_least_035",
            "standard_production_ap75_at_least_020",
            "standard_production_ap75_ap50_ratio_at_least_045",
            "standard_production_completed_task_health_floors",
            "upstream_sanity_ap_50_95_at_least_025",
            "upstream_sanity_ap50_at_least_040",
            "upstream_sanity_ap75_at_least_025",
            "upstream_sanity_ap75_ap50_ratio_at_least_050",
            "canonical_accuracy_claim",
            "_deterministic_archive",
            "_write_archive_manifest",
            "archive_member_count",
            '"status": "completed"',
            '"status": "accepted"',
        )
    ) and all(
        marker in wrapper_text
        for marker in (
            "onnx_splitpoint_tool.dependency_bootstrap",
            "--groups yolov7_probe",
            "probe_yolov7_decoder_ab.py",
        )
    ) and (
        validator_mirror.read_bytes() == validator_bytes
        and validator_sha256 == VALIDATOR_MIRROR_SHA256
        and all(
            marker in validator_text
            for marker in (
                "yolov7_paper_raw_head_requires_model_bound_decoder_contract",
                "rejected:unbound_yolov7_paper_raw_head",
                "'detections': []",
                "'claim_capable': False",
            )
        )
    )


def _non_audit_debug_pack_contract() -> bool:
    with tempfile.TemporaryDirectory(prefix="v27547-non-audit-smoke-") as tmp:
        root = Path(tmp)
        report = root / "reports/scientific/native_ranking_audit.json"
        report.parent.mkdir(parents=True)
        report.write_text(
            json.dumps({"status": "not_requested"}) + "\n",
            encoding="utf-8",
        )
        (root / "profile.yaml").write_text(
            yaml.safe_dump({
                "selection_policy": {
                    "selection_strategy": "stratified_windows",
                    "score_independent_audit_enabled": False,
                },
                "model_suite": {"primary": [{"id": "yolov7_paper"}]},
            }),
            encoding="utf-8",
        )
        audit = discover_ranking_audit_evidence(root)
        return bool(
            audit.get("enabled") is False
            and audit.get("profile_requested") is False
            and audit.get("explicit_report_present") is True
            and audit.get("model_ids") == []
            and audit.get("expected_members") == []
            and audit.get("missing_source_members") == []
            and audit.get("source_contract_ok") is True
        )


def _anchor_profile_contract(root: Path) -> bool:
    path = root / "profiles/yolov7_paper_v27547_standard_anchor_b500.yaml"
    if not path.is_file():
        return False
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    snapshot = dict(
        ((payload.get("execution_preset") or {}).get("snapshot") or {})
    )
    models = list((payload.get("model_suite") or {}).get("primary") or [])
    model = dict(models[0]) if len(models) == 1 else {}
    quality = dict(snapshot.get("quality") or {})
    native = dict((snapshot.get("runtime") or {}).get("native") or {})
    data = dict(snapshot.get("data") or {})
    official = dict(payload.get("official_coco_evaluation") or {})
    return bool(
        payload.get("name") == "yolov7_paper_v27547_standard_anchor_b500"
        and model.get("id") == "yolov7_paper"
        and model.get("model_sha256") == MODEL_SHA256
        and (payload.get("selection_policy") or {}).get("forced_cases")
        == {"yolov7_paper": ["b044"]}
        and snapshot.get("reproducibility", {}).get("verify_model_content")
        is True
        and data.get("calibration_items", {}).get("detection") == 500
        and data.get("validation_items", {}).get("detection") == 500
        and native.get("backends") == ["hailo8", "hailo10h", "deepx"]
        and native.get("frames") == 1000
        and native.get("warmup") == 100
        and native.get("repetitions") == 3
        and native.get("full_baselines") is True
        and (payload.get("execution_preset") or {}).get(
            "overrides", {}
        ).get("energy_enabled") is False
        and quality.get("cache_task_quality") is False
        and quality.get("official_coco_enabled") is True
        and quality.get("official_coco_required") is True
        and official.get("enabled") is True
        and official.get("required_for_final") is True
        and official.get("require_pycocotools") is True
        and snapshot.get("ranking", {}).get("enabled") is False
        and snapshot.get("build", {}).get("hailo", {}).get(
            "cache_integrity"
        ) == "relaxed"
        and snapshot.get("build", {}).get("artifact_store", {}).get(
            "verify_on_reuse"
        ) == "metadata"
        and snapshot.get("build", {}).get("deepx", {}).get("cache_dir")
        == (
            "~/Models/BackendArtifacts/deepx/v2.75.44/"
            "thesis_standard_b500_imagenet_mean_std"
        )
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    build = package_build_snapshot()
    critical = set(build.get("critical_module_sha256") or {})
    checks: dict[str, Any] = {
        "version": __version__ == VERSION,
        "release": __release__ == VERSION,
        "lineage": __development_lineage__ == f"v{VERSION}",
        "workflow": WORKFLOW_VERSION == BUILD_ID,
        "build_id": __build_id__ == BUILD_ID,
        "build_contract": int(__build_contract_version__) == 2,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "release_files": all((root / name).is_file() for name in REQUIRED_FILES),
        "entry_points": all(
            marker in pyproject
            for marker in (
                'version = "2.75.47"',
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
            )
        ),
        "yolov7_model_bound_decoder_and_ab_probe": (
            _yolov7_contract_sources(root)
        ),
        "debug_pack_non_audit_intent": _non_audit_debug_pack_contract(),
        "yolov7_anchor_profile": _anchor_profile_contract(root),
        "critical_inventory_complete": (
            build.get("critical_module_set_complete") is True
        ),
        "claim_critical_repairs": CRITICAL_MODULES.issubset(critical),
        "package_content_digest": str(
            build.get("package_content_sha256") or ""
        ).startswith("sha256:"),
    }
    result = {
        "schema": "onnx-splitpoint/v27547-smoke",
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
