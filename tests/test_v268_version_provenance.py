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
from onnx_splitpoint_tool.v267_smoke import main as v267_smoke_main
from onnx_splitpoint_tool.v268_smoke import REQUIRED_FEATURES, main as v268_smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v268_release_identity() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))


def test_v268_pyproject_exposes_current_and_previous_smokes() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in text
    assert 'onnx-splitpoint-smoke-v268 = "onnx_splitpoint_tool.v268_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-68 = "onnx_splitpoint_tool.v268_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-67 = "onnx_splitpoint_tool.v267_smoke:main"' in text


def test_v268_release_smoke_is_hardware_independent(capsys) -> None:
    assert v268_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v268-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0


def test_v267_smoke_alias_remains_compatible(capsys) -> None:
    assert v267_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v267-smoke"
    assert result["ok"] is True


def test_v268_build_snapshot_covers_current_claim_modules() -> None:
    build = package_build_snapshot()
    modules = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert {
        "v268_smoke.py",
        "execution_plan.py",
        "run_modes.py",
        "workflow/runner.py",
        "resources/remote_scripts/native_host_telemetry.py",
        "resources/remote_scripts/native_full_baseline_eval_runner.py",
        "resources/remote_scripts/native_producer_final_report.py",
        "resources/remote_scripts/native_trt_from_benchmarkset.py",
    }.issubset(modules)


def test_v268_release_documentation_uses_current_identity() -> None:
    for path in (
        Path("README.md"),
        Path("docs/VERSIONING.md"),
        Path("VERSION_2.75.47_BUILD_AND_TEST_REPORT.md"),
    ):
        text = path.read_text(encoding="utf-8")
        assert "2.75.47" in text
        assert "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent" in text
