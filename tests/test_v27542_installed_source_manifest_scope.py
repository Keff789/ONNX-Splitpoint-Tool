from __future__ import annotations

import fcntl
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import build_source_manifest, build_source_release


def _write_release_identity(root: Path, version: str, build_id: str) -> None:
    """Write the authoritative identity expected by the current manifest tool."""
    (root / "onnx_splitpoint_tool/release_identity.py").write_text(
        f'VERSION = "{version}"\nBUILD_ID = "{build_id}"\n',
        encoding="utf-8",
    )


CURRENT_UPDATER_REQUIRED_ENTRY_POINTS = (
    "onnx-splitpoint-smoke-v279=onnx_splitpoint_tool.v279_smoke:main",
    "onnx-splitpoint-smoke-v2791=onnx_splitpoint_tool.v279_smoke:main",
    "onnx-splitpoint-smoke-v2-79-1=onnx_splitpoint_tool.v279_smoke:main",
    "onnx-splitpoint-smoke-v2792=onnx_splitpoint_tool.v2792_smoke:main",
    "onnx-splitpoint-smoke-v2-79-2=onnx_splitpoint_tool.v2792_smoke:main",
    "onnx-splitpoint-smoke-v2793=onnx_splitpoint_tool.v2793_smoke:main",
    "onnx-splitpoint-smoke-v2-79-3=onnx_splitpoint_tool.v2793_smoke:main",
    "onnx-splitpoint-smoke-v2794=onnx_splitpoint_tool.v2794_smoke:main",
    "onnx-splitpoint-smoke-v2-79-4=onnx_splitpoint_tool.v2794_smoke:main",
    "onnx-splitpoint-smoke-v2795=onnx_splitpoint_tool.v2795_smoke:main",
    "onnx-splitpoint-smoke-v2-79-5=onnx_splitpoint_tool.v2795_smoke:main",
    "onnx-splitpoint-smoke-v2796=onnx_splitpoint_tool.v2796_smoke:main",
    "onnx-splitpoint-smoke-v2-79-6=onnx_splitpoint_tool.v2796_smoke:main",
    "onnx-splitpoint-smoke-v2797=onnx_splitpoint_tool.v2797_smoke:main",
    "onnx-splitpoint-smoke-v2-79-7=onnx_splitpoint_tool.v2797_smoke:main",
    "onnx-splitpoint-smoke-v2798=onnx_splitpoint_tool.v2798_smoke:main",
    "onnx-splitpoint-smoke-v2-79-8=onnx_splitpoint_tool.v2798_smoke:main",
    "onnx-splitpoint-smoke-v2799=onnx_splitpoint_tool.v2799_smoke:main",
    "onnx-splitpoint-smoke-v2-79-9=onnx_splitpoint_tool.v2799_smoke:main",
    "onnx-splitpoint-smoke-v27910=onnx_splitpoint_tool.v27910_smoke:main",
    "onnx-splitpoint-smoke-v2-79-10=onnx_splitpoint_tool.v27910_smoke:main",
    "onnx-splitpoint-smoke-v27911=onnx_splitpoint_tool.v27911_smoke:main",
    "onnx-splitpoint-smoke-v2-79-11=onnx_splitpoint_tool.v27911_smoke:main",
    "onnx-splitpoint-smoke-v27912=onnx_splitpoint_tool.v27912_smoke:main",
    "onnx-splitpoint-smoke-v2-79-12=onnx_splitpoint_tool.v27912_smoke:main",
    "onnx-splitpoint-smoke-v27913=onnx_splitpoint_tool.v27913_smoke:main",
    "onnx-splitpoint-smoke-v2-79-13=onnx_splitpoint_tool.v27913_smoke:main",
    "onnx-splitpoint-platform-power=onnx_splitpoint_tool.platform_power_cli:main",
    "onnx-splitpoint-smoke-v2-79=onnx_splitpoint_tool.v279_smoke:main",
    "onnx-splitpoint-backend-semantic-smoke=onnx_splitpoint_tool.backend_semantic_smoke:main",
    "onnx-splitpoint-smoke-v278=onnx_splitpoint_tool.v278_smoke:main",
    "onnx-splitpoint-smoke-v2-78=onnx_splitpoint_tool.v278_smoke:main",
    "onnx-splitpoint-smoke-v277=onnx_splitpoint_tool.v277_smoke:main",
    "onnx-splitpoint-smoke-v2-77=onnx_splitpoint_tool.v277_smoke:main",
    "onnx-splitpoint-smoke-v276=onnx_splitpoint_tool.v276_smoke:main",
    "onnx-splitpoint-smoke-v2-76=onnx_splitpoint_tool.v276_smoke:main",
    "onnx-splitpoint-smoke-v27550=onnx_splitpoint_tool.v27550_smoke:main",
    "onnx-splitpoint-smoke-v2-75-50=onnx_splitpoint_tool.v27550_smoke:main",
    "onnx-splitpoint-smoke-v27549=onnx_splitpoint_tool.v27549_smoke:main",
    "onnx-splitpoint-smoke-v2-75-49=onnx_splitpoint_tool.v27549_smoke:main",
    "onnx-splitpoint-smoke-v27548=onnx_splitpoint_tool.v27548_smoke:main",
    "onnx-splitpoint-smoke-v2-75-48=onnx_splitpoint_tool.v27548_smoke:main",
    "onnx-splitpoint-smoke-v27547=onnx_splitpoint_tool.v27547_smoke:main",
    "onnx-splitpoint-smoke-v2-75-47=onnx_splitpoint_tool.v27547_smoke:main",
    "onnx-splitpoint-smoke-v27546=onnx_splitpoint_tool.v27546_smoke:main",
    "onnx-splitpoint-smoke-v2-75-46=onnx_splitpoint_tool.v27546_smoke:main",
    "onnx-splitpoint-smoke-v27545=onnx_splitpoint_tool.v27545_smoke:main",
    "onnx-splitpoint-smoke-v2-75-45=onnx_splitpoint_tool.v27545_smoke:main",
    "onnx-splitpoint-smoke-v27544=onnx_splitpoint_tool.v27544_smoke:main",
    "onnx-splitpoint-smoke-v2-75-44=onnx_splitpoint_tool.v27544_smoke:main",
    "onnx-splitpoint-smoke-v27543=onnx_splitpoint_tool.v27543_smoke:main",
    "onnx-splitpoint-smoke-v2-75-43=onnx_splitpoint_tool.v27543_smoke:main",
)


@pytest.fixture
def release_tree(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    (root / "onnx_splitpoint_tool/workflow").mkdir(parents=True)
    (root / "scripts").mkdir()
    (root / "profiles").mkdir()
    (root / "pyproject.toml").write_text(
        '[project]\nversion = "2.75.42"\n',
        encoding="utf-8",
    )
    (root / "onnx_splitpoint_tool/workflow/runner.py").write_text(
        'WORKFLOW_VERSION = "v2.75.42-test"\n',
        encoding="utf-8",
    )
    _write_release_identity(root, "2.75.42", "v2.75.42-test")
    (root / "scripts/release_tool.py").write_text(
        'print("release")\n',
        encoding="utf-8",
    )
    (root / "profiles/release.yaml").write_text(
        "profile: release\n",
        encoding="utf-8",
    )
    build_source_manifest.build(root)
    return root


def test_release_scope_remains_strict_and_is_the_default(
    release_tree: Path,
) -> None:
    assert build_source_manifest.verify(release_tree)["ok"] is True

    (release_tree / "profiles/user.yaml").write_text(
        "profile: user\n",
        encoding="utf-8",
    )

    default_result = build_source_manifest.verify(release_tree)
    explicit_result = build_source_manifest.verify(
        release_tree,
        scope="release",
    )
    assert default_result == explicit_result
    assert default_result["ok"] is False
    assert default_result["checks"]["file_count_ok"] is False
    assert default_result["checks"]["files_ok"] is False
    # SHA256SUMS is the exact byte projection of the authoritative manifest,
    # not of an independently rescanned tree.  The unmanifested profile is
    # rejected by file_count/files while the unchanged index stays coherent.
    assert default_result["checks"]["sha256sums_ok"] is True


def test_installed_scope_reports_regular_top_level_user_profiles(
    release_tree: Path,
) -> None:
    user_yaml = release_tree / "profiles/user.yaml"
    user_yml = release_tree / "profiles/experiment.yml"
    user_yaml.write_text("profile: user\n", encoding="utf-8")
    user_yml.write_text("profile: experiment\n", encoding="utf-8")

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is True
    assert result["scope"] == "installed"
    assert all(result["checks"].values())
    assert result["release_file_count"] == result["file_count"]
    assert result["actual_allowlisted_file_count"] == result["file_count"] + 2
    assert result["user_profile_count"] == 2
    assert [row["path"] for row in result["user_profiles"]] == [
        "profiles/experiment.yml",
        "profiles/user.yaml",
    ]
    assert result["unexpected_extras"] == []
    assert result["symlinks"] == []


def test_installed_scope_matches_updater_case_insensitive_yaml_policy(
    release_tree: Path,
) -> None:
    (release_tree / "profiles/user.YAML").write_text(
        "profile: user\n",
        encoding="utf-8",
    )

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is True
    assert [row["path"] for row in result["user_profiles"]] == [
        "profiles/user.YAML",
    ]


def test_installed_scope_is_available_through_explicit_cli_option(
    release_tree: Path,
) -> None:
    (release_tree / "profiles/user.yaml").write_text(
        "profile: user\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        [
            sys.executable,
            str(Path(build_source_manifest.__file__)),
            "--root",
            str(release_tree),
            "--verify",
            "--scope",
            "installed",
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    result = json.loads(completed.stdout)
    assert result["ok"] is True
    assert result["scope"] == "installed"
    assert result["user_profile_count"] == 1


def test_installed_scope_rejects_unmanifested_source_script(
    release_tree: Path,
) -> None:
    (release_tree / "scripts/obsolete_acceptance.sh").write_text(
        "#!/usr/bin/env bash\n",
        encoding="utf-8",
    )

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is False
    assert result["checks"]["unexpected_extras_ok"] is False
    assert result["unexpected_extra_count"] == 1
    assert [row["path"] for row in result["unexpected_extras"]] == [
        "scripts/obsolete_acceptance.sh",
    ]


@pytest.mark.parametrize(
    "relative_path,is_directory",
    (
        ("profiles/user.json", False),
        ("profiles/nested", True),
        ("profiles/not-a-regular-profile.yaml", True),
    ),
)
def test_installed_scope_rejects_other_unmanifested_profile_entries(
    release_tree: Path,
    relative_path: str,
    is_directory: bool,
) -> None:
    path = release_tree / relative_path
    if is_directory:
        path.mkdir()
    else:
        path.write_text("not: admitted\n", encoding="utf-8")

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is False
    assert result["checks"]["unexpected_extras_ok"] is False
    assert [row["path"] for row in result["unexpected_extras"]] == [
        relative_path,
    ]


@pytest.mark.parametrize("mutation", ("changed", "missing"))
def test_installed_scope_rejects_changed_or_missing_release_file(
    release_tree: Path,
    mutation: str,
) -> None:
    owned = release_tree / "scripts/release_tool.py"
    if mutation == "changed":
        owned.write_text('print("changed")\n', encoding="utf-8")
    else:
        owned.unlink()

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is False
    assert result["checks"]["files_ok"] is False
    if mutation == "changed":
        assert [row["path"] for row in result["changed"]] == [
            "scripts/release_tool.py",
        ]
        assert result["missing"] == []
    else:
        assert result["changed"] == []
        assert result["missing"] == ["scripts/release_tool.py"]


def test_installed_scope_rejects_user_profile_symlink(
    release_tree: Path,
) -> None:
    os.symlink(
        "release.yaml",
        release_tree / "profiles/user.yaml",
    )

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is False
    assert result["checks"]["symlinks_ok"] is False
    assert result["user_profiles"] == []
    assert result["symlinks"] == ["profiles/user.yaml"]


def test_installed_scope_rejects_manifest_owned_symlink_even_with_same_bytes(
    release_tree: Path,
) -> None:
    owned = release_tree / "scripts/release_tool.py"
    backing = release_tree / "release_tool.py.backing"
    backing.write_bytes(owned.read_bytes())
    owned.unlink()
    os.symlink("../release_tool.py.backing", owned)

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is False
    assert result["checks"]["symlinks_ok"] is False
    assert result["checks"]["files_ok"] is False
    assert result["symlinks"] == ["scripts/release_tool.py"]
    assert result["missing"] == ["scripts/release_tool.py"]


@pytest.mark.parametrize("corruption", ("changed", "missing"))
def test_installed_scope_verifies_release_sha256sums(
    release_tree: Path,
    corruption: str,
) -> None:
    (release_tree / "profiles/user.yaml").write_text(
        "profile: user\n",
        encoding="utf-8",
    )
    sums = release_tree / "SHA256SUMS.txt"
    if corruption == "changed":
        sums.write_text(
            sums.read_text(encoding="utf-8") + "unexpected\n",
            encoding="utf-8",
        )
    else:
        sums.unlink()

    result = build_source_manifest.verify(
        release_tree,
        scope="installed",
    )

    assert result["ok"] is False
    assert result["checks"]["sha256sums_ok"] is False
    assert result["checks"]["files_ok"] is True
    assert result["user_profile_count"] == 1


def test_current_updater_offline_refresh_is_fail_closed_and_ordered() -> None:
    updater = (
        Path(__file__).resolve().parents[1]
        / "scripts/update_source_release.sh"
    ).read_text(encoding="utf-8")
    for marker in (
        "--checksum",
        "scripts/refresh_editable_install.py",
        "--expected-version 2.79.13",
        "--require-entrypoint",
        "onnx-splitpoint-smoke-v2796=",
        "onnx-splitpoint-smoke-v2-79-6=",
        "onnx-splitpoint-smoke-v2797=",
        "onnx-splitpoint-smoke-v2-79-7=",
        "onnx-splitpoint-smoke-v2798=",
        "onnx-splitpoint-smoke-v2-79-8=",
        "onnx-splitpoint-smoke-v2799=",
        "onnx-splitpoint-smoke-v2-79-9=",
        "onnx-splitpoint-smoke-v27910=",
        "onnx-splitpoint-smoke-v2-79-10=",
        "onnx-splitpoint-smoke-v27911=",
        "onnx-splitpoint-smoke-v2-79-11=",
        "onnx-splitpoint-smoke-v27912=",
        "onnx-splitpoint-smoke-v2-79-12=",
        "onnx-splitpoint-smoke-v27913=",
        "onnx-splitpoint-smoke-v2-79-13=",
        "onnx-splitpoint-platform-power=",
        "onnx-splitpoint-smoke-v27547=",
        "onnx-splitpoint-smoke-v2-75-47=",
        "onnx-splitpoint-smoke-v27546=",
        "onnx-splitpoint-smoke-v2-75-46=",
        "onnx-splitpoint-smoke-v27545=",
        "onnx-splitpoint-smoke-v2-75-45=",
        "onnx-splitpoint-smoke-v27544=",
        "onnx-splitpoint-smoke-v2-75-44=",
        "onnx-splitpoint-smoke-v27543=",
        "onnx-splitpoint-smoke-v2-75-43=",
        "--run-entrypoint onnx-splitpoint-smoke-v27913",
    ):
        assert marker in updater
    for forbidden in ("-m pip", "PIP_NO_INDEX", "setuptools.build_meta"):
        assert forbidden not in updater

    required_contracts = tuple(
        re.findall(r"'([^']+=[^']+:main)'", updater)
    )
    assert required_contracts == CURRENT_UPDATER_REQUIRED_ENTRY_POINTS

    backup_sync = updater.index("rsync -a")
    change_directory = updater.index('cd -- "$TOOL_DIR"', backup_sync)
    verify_function = updater.index("verify_installed_release()", change_directory)
    convergence_function = updater.index(
        "converge_installed_release()", verify_function
    )
    stability_function = updater.index(
        "installed_source_stable()", convergence_function
    )
    transaction = updater.index("CONVERGENCE_USED=0", stability_function)
    refresh = updater.index("scripts/refresh_editable_install.py", transaction)
    backup_block = updater[backup_sync:change_directory]
    convergence_block = updater[convergence_function:stability_function]
    stability_block = updater[stability_function:transaction]
    assert "--backup" in backup_block
    assert '--backup-dir="$BACKUP_DIR"' in backup_block
    assert "--delay-updates" not in backup_block
    assert "--delay-updates" in convergence_block
    assert "--backup" in convergence_block
    assert '--backup-dir="$BACKUP_DIR/convergence"' in convergence_block
    installed_verify_block = updater[verify_function:convergence_function]
    assert "build_source_manifest.py" in installed_verify_block
    assert '--verify --scope installed' in installed_verify_block
    assert '--verify-archive "$ZIP"' not in installed_verify_block
    assert "--dry-run" in stability_block
    assert "--itemize-changes" in stability_block
    assert "--backup" not in stability_block
    assert "if ! verify_installed_release initial" in updater[transaction:refresh]
    assert "verify_installed_release convergence" in updater[transaction:refresh]
    assert "installed_source_stable convergence" in updater[transaction:refresh]
    assert 'exec {UPDATE_LOCK_FD}<"$TOOL_DIR"' in updater
    assert 'flock -n "$UPDATE_LOCK_FD"' in updater
    assert updater.index('flock -n "$UPDATE_LOCK_FD"') < backup_sync

    assert updater.rindex('--verify-archive "$ZIP"') < refresh
    assert refresh < updater.rindex("--scope installed")


@pytest.mark.parametrize(
    "fault_mode",
    (
        "recover_first_verifier",
        "repeat_first_verifier",
        "convergence_rsync_failure",
        "recover_post_verifier_drift",
        "repeat_post_verifier_drift",
        "initial_dry_run_failure",
        "final_dry_run_failure",
    ),
)
def test_updater_repairs_same_size_same_mtime_content_and_preserves_user_state(
    tmp_path: Path,
    fault_mode: str,
) -> None:
    """Exercise checksum repair plus a deliberately incomplete first pass."""
    project_root = Path(__file__).resolve().parents[1]
    release_root = tmp_path / "release"
    (release_root / "onnx_splitpoint_tool/workflow").mkdir(parents=True)
    (release_root / "scripts").mkdir()
    (release_root / "profiles").mkdir()

    current_entries = dict(
        item.split("=", 1)
        for item in CURRENT_UPDATER_REQUIRED_ENTRY_POINTS
    )
    # The real current project retains this older launcher even though the
    # updater's required-current subset starts at v2.75.43.
    current_entries[
        "onnx-splitpoint-smoke-v27541"
    ] = "onnx_splitpoint_tool.v27541_smoke:main"
    project_scripts = "".join(
        f'{name} = "{target}"\n'
        for name, target in current_entries.items()
    )
    (release_root / "pyproject.toml").write_text(
        f"""[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "onnx-splitpoint-tool"
version = "2.79.13"

[project.scripts]
{project_scripts}

[tool.setuptools.packages.find]
include = ["onnx_splitpoint_tool*"]
""",
        encoding="utf-8",
    )
    (release_root / "onnx_splitpoint_tool/__init__.py").write_text(
        '__version__ = "2.79.13"\n',
        encoding="utf-8",
    )
    (release_root / "onnx_splitpoint_tool/workflow/runner.py").write_text(
        'WORKFLOW_VERSION = "v2.79.13-platform-power-calibration-operational-repair"\n',
        encoding="utf-8",
    )
    _write_release_identity(
        release_root,
        "2.79.13",
        "v2.79.13-platform-power-calibration-operational-repair",
    )
    for target in sorted(set(current_entries.values())):
        module_name, attribute = target.split(":", 1)
        assert attribute == "main"
        assert module_name.startswith("onnx_splitpoint_tool.")
        module_path = release_root / (module_name.replace(".", "/") + ".py")
        module_path.write_text("def main():\n    return 0\n", encoding="utf-8")
    (release_root / "profiles/release.yaml").write_text(
        "profile: release\n",
        encoding="utf-8",
    )
    for script_name in (
        "build_source_manifest.py",
        "build_source_release.py",
        "refresh_editable_install.py",
        "update_source_release.sh",
    ):
        shutil.copy2(
            project_root / "scripts" / script_name,
            release_root / "scripts" / script_name,
        )
    pristine_payloads = {
        "scripts/release_payload_a.py": b'VALUE = "AAAA"\n',
        "onnx_splitpoint_tool/release_payload_b.py": b'VALUE = "BBBB"\n',
    }
    installed_payloads = {
        "scripts/release_payload_a.py": b'VALUE = "ZZZZ"\n',
        "onnx_splitpoint_tool/release_payload_b.py": b'VALUE = "YYYY"\n',
    }
    for relative, payload in pristine_payloads.items():
        path = release_root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    between_pass_relative = "scripts/release_payload_between_pass.py"
    between_pass_original = b'VALUE = "CCCC"\n'
    (release_root / between_pass_relative).write_bytes(between_pass_original)

    build_source_manifest.build(release_root)
    archive = tmp_path / "ONNX-Splitpoint-Tool_v2.79.13_source.zip"
    release_result = build_source_release.build_release(release_root, archive)
    assert release_result["ok"] is True

    extracted_parent = tmp_path / "extracted"
    subprocess.run(
        ["unzip", "-q", str(archive), "-d", str(extracted_parent)],
        check=True,
    )
    extracted = extracted_parent / str(release_result["prefix"])
    update_home = tmp_path / "home with space"
    installed = update_home / "ONNX-Splitpoint-Tool"
    shutil.copytree(extracted, installed)

    venv_root = installed / ".venv"
    base_python = Path(
        getattr(sys, "_base_executable", sys.executable),
    ).resolve()
    subprocess.run(
        [
            str(base_python),
            "-m",
            "venv",
            str(venv_root),
        ],
        check=True,
    )
    venv_python = venv_root / "bin/python"
    isolated_probe = subprocess.run(
        [
            str(venv_python),
            "-I",
            "-c",
            (
                "import importlib.util; "
                "assert importlib.util.find_spec('pip') is not None; "
                "assert importlib.util.find_spec('setuptools') is None"
            ),
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert isolated_probe.returncode == 0, (
        isolated_probe.stdout + isolated_probe.stderr
    )

    legacy_source = tmp_path / "legacy-v27541-source"
    (legacy_source / "onnx_splitpoint_tool").mkdir(parents=True)
    (legacy_source / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "onnx-splitpoint-tool"
version = "2.75.41"

[project.scripts]
onnx-splitpoint-smoke-v27541 = "onnx_splitpoint_tool.v27541_smoke:main"

[tool.setuptools.packages.find]
include = ["onnx_splitpoint_tool*"]
""",
        encoding="utf-8",
    )
    (legacy_source / "onnx_splitpoint_tool/__init__.py").write_text(
        '__version__ = "2.75.41"\n',
        encoding="utf-8",
    )
    (legacy_source / "onnx_splitpoint_tool/v27541_smoke.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )

    dependency_source = tmp_path / "preserved-helper-source"
    (dependency_source / "preserved_helper").mkdir(parents=True)
    (dependency_source / "pyproject.toml").write_text(
        """[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "preserved-helper"
version = "9.1.0"

[tool.setuptools.packages.find]
include = ["preserved_helper*"]
""",
        encoding="utf-8",
    )
    (dependency_source / "preserved_helper/__init__.py").write_text(
        '__version__ = "9.1.0"\n',
        encoding="utf-8",
    )

    refresher = project_root / "scripts/refresh_editable_install.py"

    def refresh_source(
        source: Path,
        version: str,
        *required_entry_points: str,
    ) -> None:
        command = [
            str(venv_python),
            "-B",
            str(refresher),
            "--root",
            str(source),
            "--expected-version",
            version,
        ]
        for required in required_entry_points:
            command.extend(("--require-entrypoint", required))
        refreshed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ""},
            cwd=tmp_path,
        )
        assert refreshed.returncode == 0, refreshed.stdout + refreshed.stderr

    refresh_source(dependency_source, "9.1.0")
    refresh_source(
        legacy_source,
        "2.75.41",
        (
            "onnx-splitpoint-smoke-v27541="
            "onnx_splitpoint_tool.v27541_smoke:main"
        ),
    )

    def distribution_snapshot() -> dict[str, object]:
        probe = subprocess.run(
            [
                str(venv_python),
                "-B",
                "-c",
                (
                    "import importlib.metadata as m, json; "
                    "d=m.distribution('onnx-splitpoint-tool'); "
                    "print(json.dumps({"
                    "'version':d.version,"
                    "'entries':sorted(e.name for e in d.entry_points "
                    "if e.group=='console_scripts'),"
                    "'helper':m.version('preserved-helper')}))"
                ),
            ],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ""},
            cwd=tmp_path,
        )
        assert probe.returncode == 0, probe.stdout + probe.stderr
        return json.loads(probe.stdout)

    before_update = distribution_snapshot()
    assert before_update == {
        "version": "2.75.41",
        "entries": ["onnx-splitpoint-smoke-v27541"],
        "helper": "9.1.0",
    }
    site_roots = list(venv_root.glob("lib/python*/site-packages"))
    assert len(site_roots) == 1
    site_root = site_roots[0]
    helper_metadata = next(
        site_root.glob("preserved_helper-9.1.0.dist-info/METADATA")
    )
    helper_metadata_before = helper_metadata.read_bytes()

    # Reproduce two additional legacy shapes seen across editable installers:
    # duplicate canonical egg-info and a RECORD-owned setuptools finder.
    legacy_egg_info = site_root / "onnx_splitpoint_tool-legacy.egg-info"
    legacy_egg_info.mkdir()
    (legacy_egg_info / "PKG-INFO").write_text(
        "Metadata-Version: 2.1\n"
        "Name: onnx-splitpoint-tool\n"
        "Version: 2.75.40\n\n",
        encoding="utf-8",
    )
    legacy_finder = (
        site_root / "__editable___onnx_splitpoint_tool_2_75_41_finder.py"
    )
    legacy_finder.write_text("MAPPING = {}\n", encoding="utf-8")
    legacy_dist_info = next(
        site_root.glob("onnx_splitpoint_tool-2.75.41.dist-info")
    )
    with (legacy_dist_info / "RECORD").open(
        "a",
        encoding="utf-8",
        newline="",
    ) as record:
        record.write(f"{legacy_finder.name},,\n")

    assert (venv_root / "bin/onnx-splitpoint-smoke-v27541").is_file()
    assert not (venv_root / "bin/onnx-splitpoint-smoke-v27542").exists()
    venv_sentinel = venv_root / "preserved-user-state.txt"
    venv_sentinel.write_text("keep venv user state\n", encoding="utf-8")

    for relative, corrupt_payload in installed_payloads.items():
        source = extracted / relative
        target = installed / relative
        source_stat = source.stat()
        assert len(corrupt_payload) == len(pristine_payloads[relative])
        target.write_bytes(corrupt_payload)
        os.utime(
            target,
            ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
        )
        assert target.stat().st_size == source_stat.st_size
        assert target.stat().st_mtime_ns == source_stat.st_mtime_ns
        assert target.read_bytes() != source.read_bytes()

    custom_profile = installed / "profiles/custom-user.YAML"
    custom_profile.write_text("profile: custom-user\n", encoding="utf-8")
    stale_script = installed / "scripts/obsolete_release_script.sh"
    stale_script.write_text("#!/usr/bin/env bash\nexit 99\n", encoding="utf-8")

    # The production failure that motivated the convergence pass returned a
    # successful first rsync while leaving one source payload behind.  A
    # PATH-local shim mutates an initially pristine, therefore not initially
    # backed-up file after the real backup pass.  The first authoritative
    # verifier must fail; the single convergence retry must preserve those
    # injected bytes separately and repair the hybrid tree before refresh.
    real_rsync = shutil.which("rsync")
    assert real_rsync is not None
    rsync_shim_dir = tmp_path / "rsync-shim"
    rsync_shim_dir.mkdir()
    rsync_shim = rsync_shim_dir / "rsync"
    rsync_shim.write_text(
        """#!/usr/bin/env bash
set -Eeuo pipefail
call=0
if [[ -f "$V2797_RSYNC_COUNTER" ]]; then
  read -r call < "$V2797_RSYNC_COUNTER"
fi
call=$((call + 1))
printf '%s\\n' "$call" > "$V2797_RSYNC_COUNTER"
{
  printf 'CALL=%s\\n' "$call"
  printf 'ARG=%s\\n' "$@"
} >> "$V2797_RSYNC_LOG"
if [[ "$V2797_RSYNC_FAULT_MODE" == post_* && \
      ( "$call" -eq 2 || \
        ( "$V2797_RSYNC_FAULT_MODE" == "post_repeat" && "$call" -eq 4 ) ) ]]; then
  cp -- "$V2797_RSYNC_STALE_COPY" "$V2797_RSYNC_STALE_TARGET"
fi
if [[ "$V2797_RSYNC_FAULT_MODE" == "convergence_failure" && "$call" -eq 2 ]]; then
  exit 42
fi
if [[ "$V2797_RSYNC_FAULT_MODE" == "dry_initial_failure" && "$call" -eq 2 ]]; then
  exit 43
fi
if [[ "$V2797_RSYNC_FAULT_MODE" == "dry_final_failure" && "$call" -eq 3 ]]; then
  exit 43
fi
"$V2797_REAL_RSYNC" "$@"
if [[ ( "$V2797_RSYNC_FAULT_MODE" == first_* || \
        "$V2797_RSYNC_FAULT_MODE" == "convergence_failure" || \
        "$V2797_RSYNC_FAULT_MODE" == "dry_final_failure" ) && \
      ( "$call" -eq 1 || \
        ( "$V2797_RSYNC_FAULT_MODE" == "first_repeat" && "$call" -eq 2 ) ) ]]; then
  cp -- "$V2797_RSYNC_STALE_COPY" "$V2797_RSYNC_STALE_TARGET"
fi
""",
        encoding="utf-8",
    )
    rsync_shim.chmod(0o755)
    rsync_counter = tmp_path / "rsync-counter.txt"
    rsync_log = tmp_path / "rsync-calls.txt"
    stale_copy = tmp_path / "between-pass-mutated-payload.py"
    between_pass_payload = b'VALUE = "MMMM"\n'
    assert len(between_pass_payload) == len(between_pass_original)
    stale_copy.write_bytes(between_pass_payload)

    environment = os.environ.copy()
    environment.update({
        "HOME": str(update_home),
        "PATH": str(rsync_shim_dir) + os.pathsep + environment["PATH"],
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPATH": "",
        "V2797_REAL_RSYNC": real_rsync,
        "V2797_RSYNC_COUNTER": str(rsync_counter),
        "V2797_RSYNC_LOG": str(rsync_log),
        "V2797_RSYNC_STALE_COPY": str(stale_copy),
        "V2797_RSYNC_STALE_TARGET": str(
            installed / between_pass_relative
        ),
        "V2797_RSYNC_FAULT_MODE": {
            "recover_first_verifier": "first_recover",
            "repeat_first_verifier": "first_repeat",
            "convergence_rsync_failure": "convergence_failure",
            "recover_post_verifier_drift": "post_recover",
            "repeat_post_verifier_drift": "post_repeat",
            "initial_dry_run_failure": "dry_initial_failure",
            "final_dry_run_failure": "dry_final_failure",
        }[fault_mode],
    })
    update_command = [
        "bash",
        str(extracted / "scripts/update_source_release.sh"),
        str(archive),
        str(installed),
    ]
    lock_fd = os.open(installed, os.O_RDONLY)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        blocked = subprocess.run(
            update_command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)
    assert blocked.returncode == 75
    assert "läuft bereits ein Update" in blocked.stderr
    assert not rsync_log.exists()

    completed = subprocess.run(
        update_command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )
    rsync_calls = rsync_log.read_text(encoding="utf-8").split("CALL=")[1:]
    expected_calls = {
        "recover_first_verifier": 3,
        "repeat_first_verifier": 2,
        "convergence_rsync_failure": 2,
        "recover_post_verifier_drift": 4,
        "repeat_post_verifier_drift": 4,
        "initial_dry_run_failure": 2,
        "final_dry_run_failure": 3,
    }
    assert len(rsync_calls) == expected_calls[fault_mode]

    failing = {
        "repeat_first_verifier",
        "convergence_rsync_failure",
        "repeat_post_verifier_drift",
        "initial_dry_run_failure",
        "final_dry_run_failure",
    }
    if fault_mode in failing:
        assert completed.returncode != 0
        assert custom_profile.read_text(encoding="utf-8") == (
            "profile: custom-user\n"
        )
        assert venv_sentinel.read_text(encoding="utf-8") == (
            "keep venv user state\n"
        )
        assert not (venv_root / "bin/onnx-splitpoint-smoke-v2798").exists()
        assert not list(site_root.glob(".*.refresh-new-*"))
        assert not list(site_root.glob(".*.refresh-backup-*"))
        assert not list((venv_root / "bin").glob(".*.refresh-new-*"))
        assert not list((venv_root / "bin").glob(".*.refresh-backup-*"))
        assert "PASS source release synchronized" not in completed.stdout
        assert distribution_snapshot() == before_update
        if fault_mode not in {
            "convergence_rsync_failure",
            "initial_dry_run_failure",
        }:
            convergence_backups = list(
                (update_home / "ONNX-Splitpoint-Tool_update_backups").glob(
                    f"*/convergence/{between_pass_relative}"
                )
            )
            assert len(convergence_backups) == 1
            assert convergence_backups[0].read_bytes() == between_pass_payload
        return

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "PASS source release synchronized" in completed.stdout
    assert '"installer": "stdlib-only-pep376-editable"' in completed.stdout
    assert "Distribution aktualisiert: onnx-splitpoint-tool==2.79.13" in (
        completed.stdout
    )
    assert "Eigene Profile erhalten: 1" in completed.stdout

    assert "ARG=--backup\n" in rsync_calls[0]
    assert "ARG=--delay-updates\n" not in rsync_calls[0]
    convergence_call = 1 if fault_mode == "recover_first_verifier" else 2
    assert "ARG=--delay-updates\n" in rsync_calls[convergence_call]
    assert "ARG=--backup\n" in rsync_calls[convergence_call]
    assert "ARG=--dry-run\n" not in rsync_calls[convergence_call]
    assert "ARG=--dry-run\n" in rsync_calls[-1]
    assert "ARG=--itemize-changes\n" in rsync_calls[-1]
    assert "ARG=--backup\n" not in rsync_calls[-1]

    after_update = distribution_snapshot()
    assert after_update == {
        "version": "2.79.13",
        "entries": sorted(current_entries),
        "helper": "9.1.0",
    }
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2798").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2-79-8").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2797").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2-79-7").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2796").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2-79-6").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v279").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v27547").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v2-75-47").is_file()
    assert (venv_root / "bin/onnx-splitpoint-smoke-v27541").is_file()
    assert not legacy_egg_info.exists()
    assert not legacy_finder.exists()
    assert not list(site_root.glob(".*.refresh-new-*"))
    assert not list(site_root.glob(".*.refresh-backup-*"))
    assert not list((venv_root / "bin").glob(".*.refresh-new-*"))
    assert not list((venv_root / "bin").glob(".*.refresh-backup-*"))
    assert helper_metadata.read_bytes() == helper_metadata_before
    assert venv_sentinel.read_text(encoding="utf-8") == (
        "keep venv user state\n"
    )
    after_isolated_probe = subprocess.run(
        [
            str(venv_python),
            "-I",
            "-c",
            "import importlib.util; assert importlib.util.find_spec('setuptools') is None",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert after_isolated_probe.returncode == 0, (
        after_isolated_probe.stdout + after_isolated_probe.stderr
    )

    for relative, pristine_payload in pristine_payloads.items():
        assert (installed / relative).read_bytes() == pristine_payload
    assert (installed / between_pass_relative).read_bytes() == (
        between_pass_original
    )
    assert custom_profile.read_text(encoding="utf-8") == (
        "profile: custom-user\n"
    )
    assert not stale_script.exists()

    backup_root = update_home / "ONNX-Splitpoint-Tool_update_backups"
    for relative, corrupt_payload in installed_payloads.items():
        backups = list(backup_root.glob(f"*/{relative}"))
        assert len(backups) == 1
        assert backups[0].read_bytes() == corrupt_payload
    convergence_backups = list(
        backup_root.glob(f"*/convergence/{between_pass_relative}")
    )
    assert len(convergence_backups) == 1
    assert convergence_backups[0].read_bytes() == between_pass_payload
    assert not list(backup_root.glob(f"*/{between_pass_relative}"))
    stale_backups = list(
        backup_root.glob("*/scripts/obsolete_release_script.sh")
    )
    assert len(stale_backups) == 1
    assert stale_backups[0].read_text(encoding="utf-8") == (
        "#!/usr/bin/env bash\nexit 99\n"
    )

    verification = build_source_manifest.verify(
        installed,
        scope="installed",
    )
    assert verification["ok"] is True
    assert verification["unexpected_extras"] == []
    assert verification["changed"] == []
    assert [row["path"] for row in verification["user_profiles"]] == [
        "profiles/custom-user.YAML",
    ]
