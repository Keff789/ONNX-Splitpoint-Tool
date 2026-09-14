from __future__ import annotations

from pathlib import Path

from onnx_splitpoint_tool import (
    __build_contract_version__, __build_features__, __build_id__,
    __development_lineage__, __release__, __version__,
)
from onnx_splitpoint_tool.ranking_methods import RANKING_METHOD_IMPLEMENTATION, WORKFLOW_RANKING_METHOD
from onnx_splitpoint_tool.v279_smoke import BUILD_ID, LINEAGE, REQUIRED_FEATURES, VERSION, main as smoke_main
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_v279_exact_identity_and_freezes() -> None:
    assert VERSION == "2.79.13"
    assert LINEAGE == "v2.79"
    assert BUILD_ID == "v2.79.13-platform-power-calibration-operational-repair"
    assert (__version__, __release__, __development_lineage__, __build_id__) == (VERSION, VERSION, LINEAGE, BUILD_ID)
    assert WORKFLOW_VERSION == BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)
    assert WORKFLOW_RANKING_METHOD == "cut_bytes_only"
    assert RANKING_METHOD_IMPLEMENTATION == "v277-cut-bytes-only-workflow-freeze-1"


def test_v279_packaging_and_updater() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    updater = (ROOT / "scripts/update_source_release.sh").read_text(encoding="utf-8")
    local = (ROOT / "scripts/run_local_acceptance.sh").read_text(encoding="utf-8")
    generic = (ROOT / "scripts/run_v279_small_acceptance.sh").read_text(encoding="utf-8")
    assert 'version = "2.79.13"' in pyproject
    assert 'onnx-splitpoint-smoke-v279 = "onnx_splitpoint_tool.v279_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2796 = "onnx_splitpoint_tool.v2796_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-6 = "onnx_splitpoint_tool.v2796_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2797 = "onnx_splitpoint_tool.v2797_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-7 = "onnx_splitpoint_tool.v2797_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2798 = "onnx_splitpoint_tool.v2798_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-8 = "onnx_splitpoint_tool.v2798_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2799 = "onnx_splitpoint_tool.v2799_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-9 = "onnx_splitpoint_tool.v2799_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v27910 = "onnx_splitpoint_tool.v27910_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-10 = "onnx_splitpoint_tool.v27910_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v27911 = "onnx_splitpoint_tool.v27911_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-11 = "onnx_splitpoint_tool.v27911_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v27912 = "onnx_splitpoint_tool.v27912_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-12 = "onnx_splitpoint_tool.v27912_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v27913 = "onnx_splitpoint_tool.v27913_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2-79-13 = "onnx_splitpoint_tool.v27913_smoke:main"' in pyproject
    assert 'onnx-splitpoint-smoke-v2795 = "onnx_splitpoint_tool.v2795_smoke:main"' in pyproject
    assert '--expected-version 2.79.13' in updater
    assert '--run-entrypoint onnx-splitpoint-smoke-v27913' in updater
    assert 'onnx-splitpoint-tool==2.79.13' in updater
    assert 'bash scripts/run_v27913_small_acceptance.sh "$@"' in local
    assert 'run_v27913_small_acceptance.sh" "$@"' in generic
    assert "run_v2792_small_acceptance.sh" not in generic


def test_v279_documentation_identity() -> None:
    for relative in ("README.md", "docs/README.md", "docs/VERSIONING.md"):
        assert BUILD_ID in (ROOT / relative).read_text(encoding="utf-8")
    assert (ROOT / "docs/NATIVE_THREE_STAGE.md").is_file()


def test_v279_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert "PASS v2.79.13 smoke" in capsys.readouterr().out
