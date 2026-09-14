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
from onnx_splitpoint_tool.v273_smoke import (
    CURRENT_BUILD_ID,
    CURRENT_VERSION,
    REQUIRED_FEATURES,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v273_identity_and_feature_contract_are_exact() -> None:
    assert __version__ == CURRENT_VERSION
    assert __release__ == CURRENT_VERSION
    assert __development_lineage__ == f"v{CURRENT_VERSION}"
    assert __build_id__ == CURRENT_BUILD_ID
    assert WORKFLOW_VERSION == CURRENT_BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES.issubset(set(__build_features__))


def test_v273_metadata_and_current_documents_are_exact() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.75.47"')
    assert (
        'onnx-splitpoint-smoke-v273 = '
        '"onnx_splitpoint_tool.v273_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-73 = '
        '"onnx_splitpoint_tool.v273_smoke:main"'
    ) in pyproject
    for name in (
        "TESTANLEITUNG_2.75.47.md",
        "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md",
    ):
        text = (ROOT / name).read_text(encoding="utf-8")
        assert CURRENT_VERSION in text
        assert CURRENT_BUILD_ID in text


def test_v273_release_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v273-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0


def test_v273_critical_inventory_includes_deepx_process_fence() -> None:
    build = package_build_snapshot()
    assert build["critical_module_set_complete"] is True
    assert build["critical_module_count"] >= 165
    assert "deepx/env_status.py" in set(build["critical_module_sha256"])
    assert "workflow/run_evaluation.py" in set(
        build["critical_module_sha256"]
    )
