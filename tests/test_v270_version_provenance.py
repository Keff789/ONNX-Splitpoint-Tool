from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v270_smoke import REQUIRED_FEATURES
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v270_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))
    critical_modules = set(package_build_snapshot()["critical_module_sha256"])
    assert {
        "hailo_backend.py",
        "benchmark/services.py",
        "workflow/legacy_benchmarkset_binding.py",
        "v270_smoke.py",
    }.issubset(critical_modules)


def test_v270_package_and_smoke_aliases_are_exposed() -> None:
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    lock = Path("uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'version = "2.75.47"' in lock
    assert 'onnx-splitpoint-smoke-v270 = "onnx_splitpoint_tool.v270_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-70 = "onnx_splitpoint_tool.v270_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-69f = "onnx_splitpoint_tool.v269f_smoke:main"' in pyproject


def test_v270_release_notes_and_versioning_match() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
        Path("TESTANLEITUNG_2.75.47.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text
    versioning = Path("docs/VERSIONING.md").read_text(encoding="utf-8")
    assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in versioning
