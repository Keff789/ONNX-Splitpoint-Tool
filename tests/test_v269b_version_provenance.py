from pathlib import Path

import yaml

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool import __next_major_version__
from onnx_splitpoint_tool.v269b_smoke import (
    REQUIRED_FEATURES,
    _all_applicable_remote_source_mirrors_match,
    main as smoke_main,
)
from onnx_splitpoint_tool.protocol_freeze import build_protocol_projection
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v269b_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert __next_major_version__ == "3.0.0"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))
    assert 'version = "2.75.47"' in Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in Path("uv.lock").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'onnx-splitpoint-smoke-v2-69b = "onnx_splitpoint_tool.v269b_smoke:main"' in pyproject
    assert _all_applicable_remote_source_mirrors_match() is True


def test_v269b_docs_record_current_identity() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text


def test_v269b_release_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert '"ok": true' in capsys.readouterr().out


def test_final_protocol_template_binds_current_tool_identity_by_default() -> None:
    root = Path("onnx_splitpoint_tool/resources")
    standalone = yaml.safe_load(
        (root / "campaign_templates/protocol_freeze_spec.yaml").read_text(encoding="utf-8")
    )
    assert "release_identity" not in standalone

    profile_path = root / "evaluation_profiles/thesis_final_campaign_v1.yaml"
    profile = yaml.safe_load(profile_path.read_text(encoding="utf-8"))
    projection = build_protocol_projection(profile, profile_path=profile_path)
    assert projection["release_identity"] == {
        "release_id": __version__,
        "tool_version": __version__,
    }
