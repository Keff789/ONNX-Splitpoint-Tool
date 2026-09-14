from __future__ import annotations

from pathlib import Path

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v279_smoke, v27921_smoke
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "2.79.21"
EXPECTED_BUILD = "v2.79.21-cache-preflight-atomic-publication"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_current_identity_and_smoke_alias() -> None:
    assert package.__version__ == package.__release__ == VERSION == EXPECTED_VERSION
    assert package.__development_lineage__ == "v2.79"
    assert package.__build_id__ == BUILD_ID == WORKFLOW_VERSION == EXPECTED_BUILD
    assert v27921_smoke.VERSION == v279_smoke.VERSION == EXPECTED_VERSION
    assert v27921_smoke.BUILD_ID == v279_smoke.BUILD_ID == EXPECTED_BUILD


def test_current_entrypoints_retain_historical_aliases() -> None:
    pyproject = _read("pyproject.toml")
    assert 'version = "2.79.21"' in pyproject
    assert 'name = "onnx-splitpoint-tool"\nversion = "2.79.21"' in _read("uv.lock")
    for number in (19, 20, 21):
        for alias in (f"v279{number}", f"v2-79-{number}"):
            assert (
                f'onnx-splitpoint-smoke-{alias} = '
                f'"onnx_splitpoint_tool.v279{number}_smoke:main"'
            ) in pyproject
    for path in ("scripts/run_v279_small_acceptance.sh", "scripts/run_local_acceptance.sh"):
        assert "run_v27921_small_acceptance.sh" in _read(path)


def test_updater_exact_identity_and_retained_entrypoints() -> None:
    updater = _read("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.21"',
        f'EXPECTED_BUILD_ID = "{EXPECTED_BUILD}"',
        "ONNX-Splitpoint-Tool_v2.79.21",
        "--expected-version 2.79.21",
        "--run-entrypoint onnx-splitpoint-smoke-v27921",
    ):
        assert marker in updater
    for number in (19, 20, 21):
        assert f"onnx-splitpoint-smoke-v279{number}=onnx_splitpoint_tool.v279{number}_smoke:main" in updater


def test_release_docs_are_current_and_historical_docs_are_retained() -> None:
    for path in ("TESTANLEITUNG_2.79.21.md", "VERSION_2.79.21_BUILD_AND_TEST_REPORT.md"):
        contents = _read(path)
        assert EXPECTED_BUILD in contents
        assert "NOT_RUN" in contents
        assert "legacy_unsealed" in contents
        assert "selection_changed" in contents
    for historical in (
        "TESTANLEITUNG_2.79.19.md", "VERSION_2.79.19_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.20.md", "VERSION_2.79.20_BUILD_AND_TEST_REPORT.md",
    ):
        assert (ROOT / historical).is_file()
        assert historical in _read("scripts/build_source_manifest.py")
        assert historical in _read("scripts/update_source_release.sh")


def test_current_gate_retains_behavioral_regressions_not_old_identity_gate() -> None:
    gate = _read("scripts/run_v27921_small_acceptance.sh")
    assert "tests/test_v27921_release_closure.py" in gate
    assert "tests/test_v27921_final_selection_cache_preflight.py" in gate
    assert "tests/test_v27920_hailo_deepx_reuse.py" in gate
    assert "tests/test_v27920_trt_builder_cache_reuse.py" in gate
    assert "tests/test_v27920_release_closure.py" not in gate


def test_current_hardware_independent_smoke(capsys) -> None:
    assert v27921_smoke.main() == 0
    assert "PASS v2.79.21 smoke" in capsys.readouterr().out
