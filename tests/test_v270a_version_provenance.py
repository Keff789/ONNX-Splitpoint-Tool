from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v270a_smoke import REQUIRED_FEATURES, main as smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v270a_identity_and_package_entrypoints_are_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    lock = Path("uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'version = "2.75.47"' in lock
    assert 'onnx-splitpoint-smoke-v270a = "onnx_splitpoint_tool.v270a_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-70a = "onnx_splitpoint_tool.v270a_smoke:main"' in pyproject


def test_v270a_release_smoke_and_critical_module_set(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v270a-smoke"
    assert result["ok"] is True
    assert result["passed"] == 11
    assert result["failed"] == 0
    modules = set(package_build_snapshot()["critical_module_sha256"])
    assert {
        "v270a_smoke.py",
        "quality_service.py",
        "resources/templates/benchmark_suite.py.txt",
        "resources/templates/run_split_onnxruntime.py.txt",
        "workflow/runner.py",
    }.issubset(modules)


def test_v270a_release_documents_record_current_identity() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
        Path("TESTANLEITUNG_2.75.47.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text
