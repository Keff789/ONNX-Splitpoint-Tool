from __future__ import annotations

import json
from pathlib import Path

from onnx_splitpoint_tool import (
    __build_features__,
    __build_id__,
    __development_lineage__,
    __next_major_version__,
    __release__,
    __version__,
)
from onnx_splitpoint_tool.v269e_smoke import REQUIRED_FEATURES, main as smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v269e_identity_is_consistent() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert __next_major_version__ == "3.0.0"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))

    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in pyproject
    assert 'onnx-splitpoint-smoke-v269e = "onnx_splitpoint_tool.v269e_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-69e = "onnx_splitpoint_tool.v269e_smoke:main"' in pyproject
    assert 'version = "2.75.47"' in Path("uv.lock").read_text(encoding="utf-8")


def test_v269e_docs_record_current_identity_and_scope() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text


def test_v269e_release_smoke(capsys) -> None:
    assert smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v269e-smoke"
    assert result["ok"] is True
    assert result["passed"] == 11
    assert result["failed"] == 0


def test_v269e_build_snapshot_covers_repaired_claim_modules() -> None:
    build = package_build_snapshot()
    modules = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert {
        "v269e_smoke.py",
        "benchmark/evaluation_profiles.py",
        "gui/app.py",
        "gui/controller.py",
        "gui/panels/panel_evaluation_workflow.py",
        "resources/templates/run_split_onnxruntime.py.txt",
        "split_export_runners.py",
        "workflow/contracts.py",
        "workflow/hardware_matrix.py",
        "workflow/legacy_benchmarkset_binding.py",
        "workflow/runner.py",
        "workflow/start_snapshot.py",
    }.issubset(modules)
