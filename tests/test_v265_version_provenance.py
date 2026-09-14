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
from onnx_splitpoint_tool.v265_smoke import REQUIRED_FEATURES, main as v265_smoke_main
from onnx_splitpoint_tool.workflow.artifacts import package_build_snapshot
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


def test_v265_release_identity() -> None:
    assert __version__ == "2.75.47"
    assert __release__ == "2.75.47"
    assert __development_lineage__ == "v2.75.47"
    assert __build_id__ == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert WORKFLOW_VERSION == "v2.75.47-yolov7-decoder-contract-debug-pack-audit-intent"
    assert REQUIRED_FEATURES.issubset(set(__build_features__))


def test_v265_pyproject_exposes_current_and_legacy_smokes() -> None:
    text = Path("pyproject.toml").read_text(encoding="utf-8")
    assert 'version = "2.75.47"' in text
    assert 'onnx-splitpoint-smoke-v2-67 = "onnx_splitpoint_tool.v267_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-66 = "onnx_splitpoint_tool.v266_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v265 = "onnx_splitpoint_tool.v265_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-65 = "onnx_splitpoint_tool.v265_smoke:main"' in text
    assert 'onnx-splitpoint-smoke-v2-64 = "onnx_splitpoint_tool.v264_smoke:main"' in text


def test_v265_release_smoke_is_hardware_independent(capsys) -> None:
    assert v265_smoke_main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result["schema"] == "onnx-splitpoint/v265-smoke"
    assert result["ok"] is True
    assert result["failed"] == 0


def test_v265_build_snapshot_covers_current_correctness_modules() -> None:
    build = package_build_snapshot()
    modules = set(build["critical_module_sha256"])
    assert build["critical_module_set_complete"] is True
    assert build["critical_module_count"] >= 65
    assert {
        "v265_smoke.py",
        "native_performance_reporting.py",
        "energy/collector.py",
        "quality_service.py",
        "workflow/results.py",
        "workflow/scientific_reporting.py",
        "resources/remote_scripts/native_full_baseline_eval_runner.py",
        "resources/remote_scripts/native_producer_energy_plan.py",
    }.issubset(modules)
