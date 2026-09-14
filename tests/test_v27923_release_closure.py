from __future__ import annotations

from pathlib import Path

import pytest

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v279_smoke, v27923_smoke
from onnx_splitpoint_tool.release_identity import BUILD_CONTRACT_VERSION, BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
from onnx_splitpoint_tool import source_integrity
from scripts import build_source_manifest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "2.79.23"
EXPECTED_BUILD = "v2.79.23-native-reuse-measurement-fixes"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_current_identity_and_smoke_alias() -> None:
    assert package.__version__ == package.__release__ == VERSION == EXPECTED_VERSION
    assert package.__development_lineage__ == "v2.79"
    assert BUILD_CONTRACT_VERSION == 2
    assert package.__build_id__ == BUILD_ID == WORKFLOW_VERSION == EXPECTED_BUILD
    assert v27923_smoke.VERSION == v279_smoke.VERSION == EXPECTED_VERSION
    assert v27923_smoke.BUILD_ID == v279_smoke.BUILD_ID == EXPECTED_BUILD


def test_current_entrypoints_retain_historical_aliases() -> None:
    pyproject = _read("pyproject.toml")
    assert 'version = "2.79.23"' in pyproject
    assert 'name = "onnx-splitpoint-tool"\nversion = "2.79.23"' in _read("uv.lock")
    for number in (19, 20, 21, 22, 23):
        for alias in (f"v279{number}", f"v2-79-{number}"):
            assert (
                f'onnx-splitpoint-smoke-{alias} = '
                f'"onnx_splitpoint_tool.v279{number}_smoke:main"'
            ) in pyproject
    for path in ("scripts/run_v279_small_acceptance.sh", "scripts/run_local_acceptance.sh"):
        assert "run_v27923_small_acceptance.sh" in _read(path)


def test_updater_exact_identity_and_retained_entrypoints() -> None:
    updater = _read("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.23"',
        f'EXPECTED_BUILD_ID = "{EXPECTED_BUILD}"',
        "ONNX-Splitpoint-Tool_v2.79.23",
        "--expected-version 2.79.23",
        "--run-entrypoint onnx-splitpoint-smoke-v27923",
    ):
        assert marker in updater
    for number in (19, 20, 21, 22, 23):
        assert f"onnx-splitpoint-smoke-v279{number}=onnx_splitpoint_tool.v279{number}_smoke:main" in updater


def test_release_docs_are_current_and_historical_docs_are_retained() -> None:
    for path in ("TESTANLEITUNG_2.79.23.md", "VERSION_2.79.23_BUILD_AND_TEST_REPORT.md"):
        contents = _read(path)
        assert EXPECTED_BUILD in contents
        assert "NOT_RUN" in contents
        assert "COMPILE_INFEASIBLE" in contents
        assert "TRANSIENT_INFRASTRUCTURE" in contents
    for historical in (
        "TESTANLEITUNG_2.79.19.md", "VERSION_2.79.19_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.20.md", "VERSION_2.79.20_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.21.md", "VERSION_2.79.21_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.22.md", "VERSION_2.79.22_BUILD_AND_TEST_REPORT.md",
    ):
        assert (ROOT / historical).is_file()
        assert historical in _read("scripts/build_source_manifest.py")
        assert historical in _read("scripts/update_source_release.sh")


def test_current_gate_retains_behavioral_regressions_not_old_identity_gate() -> None:
    gate = _read("scripts/run_v27923_small_acceptance.sh")
    assert "tests/test_v27923_release_closure.py" in gate
    for test in (
        "tests/test_v27923_trt_legacy_reuse.py",
        "tests/test_v27923_fs_calibration_gate.py",
        "tests/test_v27923_decoded_completion.py",
        "tests/test_v27923_native_failures.py",
        "tests/test_v27923_deepx_preflight.py",
        "tests/test_v27923_workflow_fixes.py",
        "tests/test_v27922_build_evidence_store.py",
        "tests/test_v27922_negative_backend.py",
        "tests/test_v27922_build_evidence_recovery.py",
        "tests/test_v27922_negative_preflight.py",
        "tests/test_v27922_build_context.py",
        "tests/test_v27922_auxiliary_context.py",
        "tests/test_v2783_build_evidence.py",
        "tests/test_v2783_hailo8_first_feasibility.py",
        "tests/test_v2783_hailo8_evidence_binding.py",
        "tests/test_v27921_final_selection_cache_preflight.py",
    ):
        assert test in gate
    assert "tests/test_v27922_release_closure.py" not in gate
    assert "tests/test_v27921_release_closure.py" not in gate
    assert "tests/test_v27920_hailo_deepx_reuse.py" in gate
    assert "tests/test_v27920_trt_builder_cache_reuse.py" in gate
    assert "tests/test_v27920_release_closure.py" not in gate


def test_current_hardware_independent_smoke(capsys) -> None:
    assert v27923_smoke.main() == 0
    assert "PASS v2.79.23 smoke" in capsys.readouterr().out


def _minimal_release(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    files = {
        "pyproject.toml": f'[project]\nname = "onnx-splitpoint-tool"\nversion = "{VERSION}"\n',
        "onnx_splitpoint_tool/release_identity.py": f'VERSION = "{VERSION}"\nBUILD_ID = "{BUILD_ID}"\n',
        "onnx_splitpoint_tool/runtime.py": "VALUE = 1\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    build_source_manifest.build(root)
    return root


@pytest.mark.parametrize("name", ("artifact_store", "build_evidence"))
def test_local_artifacts_and_negative_evidence_are_operational_state(
    tmp_path: Path, name: str,
) -> None:
    root = _minimal_release(tmp_path)
    store = root / name
    store.mkdir()
    evidence = store / "preserved.bin"
    evidence.write_bytes(b"compiled artifact or exact negative evidence")
    (store / "current").symlink_to(evidence.name)

    assert build_source_manifest.verify(root, scope="installed")["ok"]
    assert source_integrity.verify_installed_source_integrity(root)["ok"]
    assert not build_source_manifest.verify(root, scope="release")["ok"]
    assert evidence.read_bytes() == b"compiled artifact or exact negative evidence"

    # The exception is only for these real top-level operational directories.
    (root / "onnx_splitpoint_tool/runtime.py").write_text("VALUE = 2\n")
    assert not build_source_manifest.verify(root, scope="installed")["ok"]
    assert not source_integrity.verify_installed_source_integrity(root)["ok"]


@pytest.mark.parametrize("name", ("artifact_store", "build_evidence"))
@pytest.mark.parametrize("kind", ("symlink", "regular_file"))
def test_operational_root_name_does_not_admit_links_or_files(
    tmp_path: Path, name: str, kind: str,
) -> None:
    root = _minimal_release(tmp_path)
    candidate = root / name
    if kind == "symlink":
        candidate.symlink_to(root / "onnx_splitpoint_tool", target_is_directory=True)
    else:
        candidate.write_bytes(b"not a directory")
    assert not build_source_manifest.verify(root, scope="installed")["ok"]
    assert not source_integrity.verify_installed_source_integrity(root)["ok"]


def test_updater_protects_local_evidence_during_sync_retry_and_drift_check() -> None:
    updater = _read("scripts/update_source_release.sh")
    for name in ("artifact_store", "build_evidence"):
        assert updater.count(f"--exclude='/{name}/'") == 3
        assert name in build_source_manifest.PRESERVED_INSTALL_ROOTS
        assert name in source_integrity._PRESERVED_INSTALL_ROOTS
    assert "--exclude='*.hef'" not in updater
