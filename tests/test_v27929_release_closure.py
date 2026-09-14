from __future__ import annotations

from pathlib import Path

import pytest

import onnx_splitpoint_tool as package
from onnx_splitpoint_tool import v279_smoke, v27929_smoke
from onnx_splitpoint_tool.release_identity import BUILD_CONTRACT_VERSION, BUILD_ID, VERSION
from onnx_splitpoint_tool.workflow.runner import WORKFLOW_VERSION
from onnx_splitpoint_tool import source_integrity
from scripts import build_source_manifest


ROOT = Path(__file__).resolve().parents[1]
EXPECTED_VERSION = "2.79.29"
EXPECTED_BUILD = "v2.79.29-deepx-prepared-full-without-opencv"


def _read(relative: str) -> str:
    return (ROOT / relative).read_text(encoding="utf-8")


def test_current_identity_and_smoke_alias() -> None:
    assert package.__version__ == package.__release__ == VERSION == EXPECTED_VERSION
    assert package.__development_lineage__ == "v2.79"
    assert BUILD_CONTRACT_VERSION == 2
    assert package.__build_id__ == BUILD_ID == WORKFLOW_VERSION == EXPECTED_BUILD
    assert v27929_smoke.VERSION == v279_smoke.VERSION == EXPECTED_VERSION
    assert v27929_smoke.BUILD_ID == v279_smoke.BUILD_ID == EXPECTED_BUILD


def test_current_entrypoints_retain_historical_aliases() -> None:
    pyproject = _read("pyproject.toml")
    assert 'version = "2.79.29"' in pyproject
    assert 'name = "onnx-splitpoint-tool"\nversion = "2.79.29"' in _read("uv.lock")
    for number in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29):
        for alias in (f"v279{number}", f"v2-79-{number}"):
            assert (
                f'onnx-splitpoint-smoke-{alias} = '
                f'"onnx_splitpoint_tool.v279{number}_smoke:main"'
            ) in pyproject
    for path in ("scripts/run_v279_small_acceptance.sh", "scripts/run_local_acceptance.sh"):
        assert "run_v27929_small_acceptance.sh" in _read(path)


def test_updater_exact_identity_and_retained_entrypoints() -> None:
    updater = _read("scripts/update_source_release.sh")
    for marker in (
        'EXPECTED_VERSION = "2.79.29"',
        f'EXPECTED_BUILD_ID = "{EXPECTED_BUILD}"',
        "ONNX-Splitpoint-Tool_v2.79.29",
        "--expected-version 2.79.29",
        "--run-entrypoint onnx-splitpoint-smoke-v27929",
    ):
        assert marker in updater
    for number in (19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29):
        assert f"onnx-splitpoint-smoke-v279{number}=onnx_splitpoint_tool.v279{number}_smoke:main" in updater


def test_release_docs_are_current_and_historical_docs_are_retained() -> None:
    for path in ("TESTANLEITUNG_2.79.29.md", "VERSION_2.79.29_BUILD_AND_TEST_REPORT.md"):
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
        "TESTANLEITUNG_2.79.23.md", "VERSION_2.79.23_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.24.md", "VERSION_2.79.24_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.25.md", "VERSION_2.79.25_BUILD_AND_TEST_REPORT.md",
        "TESTANLEITUNG_2.79.26.md", "VERSION_2.79.26_BUILD_AND_TEST_REPORT.md",
    ):
        assert (ROOT / historical).is_file()
        assert historical in _read("scripts/build_source_manifest.py")
        assert historical in _read("scripts/update_source_release.sh")


def test_current_gate_retains_behavioral_regressions_not_old_identity_gate() -> None:
    gate = _read("scripts/run_v27929_small_acceptance.sh")
    assert "tests/test_v27929_release_closure.py" in gate
    for test in (
        "tests/test_v27927_result_endpoints.py",
        "tests/test_v27927_status_reporting.py",
        "tests/test_v27927_deepx_input_transfer.py",
        "tests/test_v27927_deepx_value_diagnostics.py",
        "tests/test_v27927_probe_launcher.py",

        "tests/test_v27926_deepx_full_decoded_pre_nms.py",
        "tests/test_v27926_deepx_full_downstream.py",
        "tests/test_v27926_runtime_debug_pack.py",
        "tests/test_v27926_quality_submission_failure.py",
        "tests/test_v27925_deepx_compiler_overlay.py",
        "tests/test_v27925_hailo_full_bundle_paths.py",
        "tests/test_v27925_energy_calibration_reporting.py",
        "tests/test_v27925_fast_oracle_validation.py",
        "tests/test_v27925_native_energy_raw_debug_pack.py",
        "tests/test_v27924_terminal_hailo_aliases.py",
        "tests/test_v27924_quality_reporting.py",
        "tests/test_v27924_native_error_fixes.py",
        "tests/test_v27924_trt_active_retention.py",
        "tests/test_v27924_hailo_full_portability.py",
        "tests/test_v27924_negative_preflight.py",
        "tests/test_v27924_rank3_boundary_layout.py",
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
    for required in ("score_roundoff", "probe_staging", "probe_real_process"):
        assert f"tests/test_v27928_{required}.py" in gate
    assert "tests/test_v27928_release_closure.py" not in gate
    assert "tests/test_v27929_no_opencv.py" in gate
    assert "tests/test_v27929_probe_real_process.py" in gate
    assert "tests/test_v27927_release_closure.py" not in gate
    assert "tests/test_v27926_release_closure.py" not in gate
    assert "tests/test_v27925_release_closure.py" not in gate
    assert "tests/test_v27924_release_closure.py" not in gate
    assert "tests/test_v27923_release_closure.py" not in gate
    assert "tests/test_v27922_release_closure.py" not in gate
    assert "tests/test_v27921_release_closure.py" not in gate
    assert "tests/test_v27920_hailo_deepx_reuse.py" in gate
    assert "tests/test_v27920_trt_builder_cache_reuse.py" in gate
    assert "tests/test_v27920_release_closure.py" not in gate


def test_current_hardware_independent_smoke(capsys) -> None:
    assert v27929_smoke.main() == 0
    assert "PASS v2.79.29 smoke" in capsys.readouterr().out


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
