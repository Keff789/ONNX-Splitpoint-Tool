from __future__ import annotations

import stat
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.8-3.10
    import tomli as tomllib


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "run_local_acceptance.sh"
GUIDE = ROOT / "TESTANLEITUNG_2.75.47.md"


def _source() -> str:
    return SCRIPT.read_text(encoding="utf-8")


def test_default_pytest_discovery_is_scoped_to_tests() -> None:
    metadata = tomllib.loads(
        (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    )
    assert metadata["tool"]["pytest"]["ini_options"]["testpaths"] == [
        "tests"
    ]


def test_default_discovery_ignores_embedded_legacy_source_trees(
    tmp_path: Path,
) -> None:
    """Exercise the exact collection shape that failed on Smartmirror2."""
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_current.py").write_text(
        "def test_current():\n    assert True\n",
        encoding="utf-8",
    )
    legacy = tmp_path / "work_v59" / "scripts"
    legacy.mkdir(parents=True)
    (legacy / "test_energy_measurement.py").write_text(
        "raise RuntimeError('legacy tree must not be collected')\n",
        encoding="utf-8",
    )
    (tmp_path / "pyproject.toml").write_text(
        '[tool.pytest.ini_options]\ntestpaths = ["tests"]\n',
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            "-B",
            "-m",
            "pytest",
            "--collect-only",
            "-q",
            "-p",
            "no:cacheprovider",
        ],
        cwd=tmp_path,
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stderr
    assert "tests/test_current.py::test_current" in completed.stdout
    assert "test_energy_measurement" not in completed.stdout
    assert "1 test collected" in completed.stdout


def test_acceptance_is_executable_and_cannot_exit_the_calling_shell() -> None:
    mode = SCRIPT.stat().st_mode
    assert mode & stat.S_IXUSR

    completed = subprocess.run(
        [
            "bash",
            "-c",
            'source "$1"; printf "CALLER_STILL_RUNNING\\n"',
            "acceptance-source-test",
            str(SCRIPT),
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout == "CALLER_STILL_RUNNING\n"

    guide = GUIDE.read_text(encoding="utf-8")
    assert "Final Quality" in guide
    assert "run_v27528_*_final_canary.sh" in guide
    assert "keinen Pflicht-Canary" in guide
    assert "exit 0" not in guide
    assert "logout" not in guide.casefold()


def test_release_plan_only_verifies_manifest_and_never_regenerates_it(
) -> None:
    source = _source()
    invocations = source.split('"$PY" -B scripts/build_source_manifest.py')[1:]
    assert len(invocations) == 3
    assert all("--verify" in invocation.split("\n\n", 1)[0] for invocation in invocations)
    assert "build_source_manifest.py --root" not in source
    early_verify = source.index(
        'stage "1b/10 - Early read-only source-manifest verification"'
    )
    first_pytest = source.index('pytest_gate "2/10')
    assert early_verify < first_pytest


def test_clean_archive_is_rebuilt_by_the_extracted_tree() -> None:
    source = _source()
    assert (
        'archive_c="$release_tmp/'
        'ONNX-Splitpoint-Tool_v2.75.47_source_c.zip"'
        in source
    )
    assert source.count('cd -- "$extracted"') >= 2
    assert (
        '"$PY" -B scripts/build_source_release.py \\\n'
        '      --root "$extracted" \\\n'
        '      --out "$archive_c"'
        in source
    )
    assert (
        '"$PY" -B scripts/build_source_release.py \\\n'
        '      --root "$extracted" \\\n'
        '      --verify-archive "$archive_c"'
        in source
    )
    assert 'cmp --silent "$archive_a" "$archive_b"' in source
    assert 'cmp --silent "$archive_a" "$archive_c"' in source
    assert 'sha256sum "$archive_a" "$archive_b" "$archive_c"' in source


def test_updater_integration_uses_a_real_preserved_virtual_environment() -> None:
    source = _source()
    assert "run_local_acceptance requires PY from a virtual environment" in source
    assert 'cp -a "$source_venv/." "$update_target/.venv/"' in source
    assert 'ln -s "$PY" "$update_target/.venv/bin/python"' not in source
    assert source.index('cp -a "$source_venv/." "$update_target/.venv/"') < (
        source.index('bash "$extracted/scripts/update_source_release.sh"')
    )


def test_every_pytest_invocation_has_an_explicit_tests_target(
) -> None:
    source = _source()
    assert "onnx_splitpoint_tool.v27547_smoke" in source
    assert "onnx_splitpoint_tool.v272_smoke" not in source
    assert (
        '"8/10 - Complete suite vs frozen v2.73.8 failure baseline" tests'
        in source
    )
    assert "full_suite_regression_gate" in source
    assert "v2738_full_suite_baseline_failures.txt" in source
    assert "the complete suite is not labelled green" in source
    assert '"$PY" -B -m pytest -q -p no:cacheprovider \\' in source
    assert 'pytest -q -p no:cacheprovider\n' not in source
    assert source.count("tests/test_v2735_standard_execution_path.py") == 2
    assert source.count("tests/test_v2736_single_writer_cancellation.py") == 2
    assert source.count("tests/test_p01_runner_gui_guards.py") == 2
    assert source.count("tests/test_v2736_remote_process_lease.py") == 2
    assert source.count("tests/test_v2736_variant_nested_ssh.py") == 2
    assert source.count("tests/test_v2736_native_energy_remote_lease.py") == 2
    assert source.count("tests/test_frozen_audit_generation_scope.py") == 2
    assert source.count("scripts/run_v27547_small_acceptance.sh") == 2
    assert "scripts/run_v27546_small_acceptance.sh" in source
    assert source.count(
        "tests/test_p02_atomic_stage_energy_checkpoints.py"
    ) == 2
    assert source.count(
        "tests/test_p03_native_energy_planner_invariant.py"
    ) == 2
    assert source.count("tests/test_v2713_resume_artifact_contract.py") == 2


def test_v27539_acceptance_seals_remote_and_rowless_quality_regressions() -> None:
    local = _source()
    small = (
        ROOT / "scripts" / "run_v27539_small_acceptance.sh"
    ).read_text(encoding="utf-8")
    for target in (
        "tests/test_v27539_remote_premutation_dispatch.py",
        "tests/test_v27539_remote_dispatch_propagation.py",
        "tests/test_v27538_dashboard_full_only_quality_health.py",
        "tests/test_v27538_remote_rowless_quality_status.py",
        "tests/test_v27538_aggregate_full_only_quality_status.py",
        "tests/test_v269d_management_trt_quality_summary.py",
    ):
        # Source-tree focus and Fresh Extract must both execute every repair.
        assert local.count(target) == 2
        assert small.count(target) == 1


def test_hardware_is_a_safe_omitted_local_branch() -> None:
    completed = subprocess.run(
        ["bash", str(SCRIPT), "--plan"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    assert "hardware gates: SKIP" in completed.stdout
    assert "YOLO26_FULL_QUALITY_GATE" not in _source()
    assert "NATIVE_ENERGY_MICRO_GATE" not in _source()
    assert "run_v2732_hardware_acceptance" not in _source()
