from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__,
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v27540_smoke import (
    BUILD_ID,
    CURRENT_BUILD_ID,
    CURRENT_VERSION,
    REQUIRED_FEATURES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v27540_exact_release_identity_and_features() -> None:
    assert VERSION == "2.75.40"
    assert BUILD_ID == (
        "v2.75.40-deepx-preprocessing-ab-and-full-only-quality-export-repair"
    )
    assert __version__ == CURRENT_VERSION
    assert __release__ == CURRENT_VERSION
    assert __development_lineage__ == f"v{CURRENT_VERSION}"
    assert __build_id__ == CURRENT_BUILD_ID
    assert WORKFLOW_VERSION == CURRENT_BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)


def test_v27540_packaging_metadata_is_compatible() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'version = "2.75.47"' in lock


def test_v27540_release_entry_points_profiles_and_harness() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert (
        'onnx-splitpoint-smoke-v27540 = '
        '"onnx_splitpoint_tool.v27540_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-75-40 = '
        '"onnx_splitpoint_tool.v27540_smoke:main"'
    ) in pyproject
    for name in (
        "resnet50_v27540_deepx_preprocess_a_current_scale_only.yaml",
        "resnet50_v27540_deepx_preprocess_b_imagenet_mean_std.yaml",
    ):
        assert (ROOT / "profiles" / name).is_file()
    harness = ROOT / "scripts" / "run_v27540_small_acceptance.sh"
    assert harness.is_file()
    assert harness.stat().st_mode & 0o100
    source = harness.read_text(encoding="utf-8")
    assert '[[ "${BASH_SOURCE[0]}" != "$0" ]]' in source
    assert "--verify" in source
    for test_name in (
        "test_v269e_quality_export_vendored.py",
        "test_deepx_full_preprocessing_ab_contract.py",
        "test_v27540_deepx_preprocessing_probe.py",
        "test_v27540_deepx_preprocessing_ab_pairing.py",
        "test_v27540_ready_deepx_preprocessing_ab_profiles.py",
    ):
        assert test_name in source


def test_v27540_preserves_packaged_historical_v27539_artifacts() -> None:
    # Root-level guides and build reports are current-release-only source
    # artifacts.  Historical runnable assets remain packaged for compatibility.
    for path in (
        ROOT / "profiles" / "resnet50_v27539_deepx_full_quality_canary.yaml",
        ROOT / "scripts" / "run_v27539_small_acceptance.sh",
    ):
        assert path.is_file(), path
        assert "2.75.39" in path.read_text(encoding="utf-8")


def test_v27540_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    captured = capsys.readouterr().out
    assert '"ok": true' in captured
    assert '"schema": "onnx-splitpoint/v27540-smoke"' in captured
