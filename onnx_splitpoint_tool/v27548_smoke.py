"""Compact, hardware-independent release smoke for version 2.75.48."""
from __future__ import annotations

from pathlib import Path

import yaml

from . import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from .v27547_smoke import REQUIRED_FEATURES as V27547_REQUIRED_FEATURES
from .workflow.runner import WORKFLOW_VERSION


VERSION = "2.75.48"
BUILD_ID = "v2.75.48-yolov7-anchor-closeout"
NEW_FEATURES = {
    "canonical_sha256_provenance_comparison",
    "yolov7_completion_registry_model_binding",
    "deepx_full_optional_model_identity_binding",
    "phase1_official_coco_scope_closure",
}
REQUIRED_FEATURES = set(V27547_REQUIRED_FEATURES) | NEW_FEATURES
MODEL_SHA256 = (
    "7a13e66f91047cce0e251c05f6415964"
    "6847e842af31d60441c63dcdfad7825d"
)


def _profile_is_bounded(root: Path) -> bool:
    path = root / "profiles/yolov7_paper_v27548_standard_anchor_b500.yaml"
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    snapshot = dict(
        ((payload.get("execution_preset") or {}).get("snapshot") or {})
    )
    quality = dict(snapshot.get("quality") or {})
    native = dict((snapshot.get("runtime") or {}).get("native") or {})
    data = dict(snapshot.get("data") or {})
    model = dict(((payload.get("model_suite") or {}).get("primary") or [{}])[0])
    official = dict(payload.get("official_coco_evaluation") or {})
    return bool(
        payload.get("name") == "yolov7_paper_v27548_standard_anchor_b500"
        and model.get("id") == "yolov7_paper"
        and model.get("model_sha256") == MODEL_SHA256
        and (payload.get("selection_policy") or {}).get("forced_cases")
        == {"yolov7_paper": ["b044"]}
        and data.get("validation_items", {}).get("detection") == 500
        and native.get("backends") == ["hailo8", "hailo10h", "deepx"]
        and native.get("full_baselines") is True
        and snapshot.get("ranking", {}).get("enabled") is False
        and (payload.get("execution_preset") or {}).get(
            "overrides", {}
        ).get("energy_enabled") is False
        and quality.get("cache_task_quality") is False
        and quality.get("official_coco_enabled") is False
        and quality.get("official_coco_required") is False
        and official.get("enabled") is False
        and official.get("required_for_final") is True
    )


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    required_files = (
        "profiles/yolov7_paper_v27548_standard_anchor_b500.yaml",
        "onnx_splitpoint_tool/v27548_smoke.py",
        "scripts/run_v27548_small_acceptance.sh",
        "scripts/probe_yolov7_decoder_ab.py",
        "tests/test_v27548_official_coco_sha.py",
    )
    checks = {
        "version": __version__ == VERSION,
        "release": __release__ == VERSION,
        "lineage": __development_lineage__ == f"v{VERSION}",
        "build": __build_id__ == BUILD_ID and WORKFLOW_VERSION == BUILD_ID,
        "features": REQUIRED_FEATURES.issubset(set(__build_features__)),
        "files": all((root / name).is_file() for name in required_files),
        "entrypoints": all(
            marker in pyproject
            for marker in (
                'version = "2.75.48"',
                (
                    "onnx-splitpoint-smoke-v27548 = "
                    '"onnx_splitpoint_tool.v27548_smoke:main"'
                ),
                (
                    "onnx-splitpoint-smoke-v2-75-48 = "
                    '"onnx_splitpoint_tool.v27548_smoke:main"'
                ),
            )
        ),
        "bounded_anchor_profile": _profile_is_bounded(root),
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        print("FAIL v2.75.48 smoke: " + ", ".join(failed))
        return 1
    print("PASS v2.75.48 smoke")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
