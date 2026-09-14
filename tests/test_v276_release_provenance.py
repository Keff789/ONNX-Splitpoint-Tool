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
from onnx_splitpoint_tool.v276_smoke import (
    BUILD_ID,
    LINEAGE,
    REQUIRED_FEATURES,
    VERSION,
    main as smoke_main,
)
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]


def test_v276_exact_release_identity() -> None:
    assert VERSION == "2.76.2"
    assert LINEAGE == "v2.76"
    assert BUILD_ID == "v2.76.2-normalized-cross-runner-alias-closure"
    assert __version__ == VERSION
    assert __release__ == VERSION
    assert __development_lineage__ == LINEAGE
    assert __build_id__ == BUILD_ID
    assert WORKFLOW_VERSION == BUILD_ID
    assert __build_contract_version__ == 2
    assert REQUIRED_FEATURES <= set(__build_features__)


def test_v276_packaging_and_entrypoints() -> None:
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'version = "2.76.2"' in pyproject
    root_block = lock.split('name = "onnx-splitpoint-tool"', 1)[1]
    assert root_block.lstrip().startswith('version = "2.76.2"')
    assert (
        'onnx-splitpoint-smoke-v276 = '
        '"onnx_splitpoint_tool.v276_smoke:main"'
    ) in pyproject
    assert (
        'onnx-splitpoint-smoke-v2-76 = '
        '"onnx_splitpoint_tool.v276_smoke:main"'
    ) in pyproject


def test_v276_hardware_independent_smoke(capsys) -> None:
    assert smoke_main() == 0
    assert "PASS v2.76 smoke" in capsys.readouterr().out
