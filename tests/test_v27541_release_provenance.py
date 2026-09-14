from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v27541_smoke import (
    BUILD_ID,
    CRITICAL_MODULES,
    CURRENT_BUILD_ID,
    CURRENT_VERSION,
    PROFILE_NAME,
    REQUIRED_FEATURES,
    REQUIRED_FILES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v27541_exact_release_identity_and_features() -> None:
    assert VERSION == "2.75.41"
    assert BUILD_ID == "v2.75.41-deepx-calibration-500-vs-1000-canary"
    assert CURRENT_VERSION == "2.75.47"
    assert CURRENT_BUILD_ID == (
        "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    )
    assert __version__ == CURRENT_VERSION
    assert __release__ == CURRENT_VERSION
    assert __development_lineage__ == f"v{CURRENT_VERSION}"
    assert __build_id__ == CURRENT_BUILD_ID
    assert WORKFLOW_VERSION == CURRENT_BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)


def test_v27541_packaging_documents_profile_and_harness_are_pinned() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.75.47"')
    assert (
        'onnx-splitpoint-smoke-v27541 = '
        '"onnx_splitpoint_tool.v27541_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-75-41 = '
        '"onnx_splitpoint_tool.v27541_smoke:main"'
    ) in pyproject
    assert (ROOT / "profiles" / PROFILE_NAME).is_file()
    for name in REQUIRED_FILES:
        assert (ROOT / name).is_file(), name
    harness = ROOT / "scripts/run_v27541_small_acceptance.sh"
    assert harness.stat().st_mode & 0o100
    for name in (
        "README.md",
        "docs/README.md",
        "docs/VERSIONING.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert VERSION in text
    # Root guides/reports are current-release-only manifest members. Historical
    # runnable profiles, tools and tests remain packaged; v2.75.41 root docs do
    # not become hidden dependencies of a fresh v2.75.42 extract.


def test_v27541_claim_critical_inventory_covers_both_pairing_layers() -> None:
    build = package_build_snapshot()
    critical = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert build["critical_module_count"] == build["critical_module_expected_count"]
    assert CRITICAL_MODULES <= critical


def test_v27541_preserves_historical_v27540_experiment_assets() -> None:
    for name in (
        "resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml",
        "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml",
    ):
        assert (ROOT / "profiles" / name).is_file()
    for name in (
        "scripts/verify_v27540_deepx_preprocessing_ab.py",
        "scripts/run_v27540_small_acceptance.sh",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert "2.75.40" in text


def test_v27541_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v27541-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0
