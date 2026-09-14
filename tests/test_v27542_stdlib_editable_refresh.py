from __future__ import annotations

import ast
import csv
import io
import os
from pathlib import Path

import pytest

from scripts import refresh_editable_install as refresh


ROOT = Path(__file__).resolve().parents[1]


def _minimal_project(root: Path, *, version: str = "2.75.42") -> Path:
    (root / "onnx_splitpoint_tool").mkdir(parents=True)
    (root / "pyproject.toml").write_text(
        f'''[build-system]
requires = ["setuptools>=61"]
build-backend = "setuptools.build_meta"

[project]
name = "onnx-splitpoint-tool"
version = "{version}"
dependencies = [
  "preserved-helper>=1",
]

[project.scripts]
onnx-splitpoint-smoke-v27542 = "onnx_splitpoint_tool.smoke:main"
''',
        encoding="utf-8",
    )
    (root / "onnx_splitpoint_tool/__init__.py").write_text(
        f'__version__ = "{version}"\n',
        encoding="utf-8",
    )
    (root / "onnx_splitpoint_tool/smoke.py").write_text(
        "def main():\n    return 0\n",
        encoding="utf-8",
    )
    return root


def _record_text(rows: list[tuple[str, str, str]]) -> str:
    output = io.StringIO(newline="")
    csv.writer(output, lineterminator="\n").writerows(rows)
    return output.getvalue()


def test_refresher_parses_current_project_without_tomllib_or_backend() -> None:
    source = (ROOT / "scripts/refresh_editable_install.py").read_text(
        encoding="utf-8",
    )
    ast.parse(source, feature_version=(3, 8))

    project = refresh.load_project(ROOT)
    assert project.name == "onnx-splitpoint-tool"
    # This test exercises the historical stdlib parser against the current
    # release tree.  Historical console-script identities below stay frozen,
    # while the project version must follow the release being packaged.
    assert project.version == "2.79.13"
    assert len(project.scripts) >= 100
    assert project.scripts["onnx-splitpoint-smoke-v27542"] == (
        "onnx_splitpoint_tool.v27542_smoke:main"
    )
    assert project.scripts["onnx-splitpoint-smoke-v27544"] == (
        "onnx_splitpoint_tool.v27544_smoke:main"
    )
    assert project.scripts["onnx-splitpoint-smoke-v27545"] == (
        "onnx_splitpoint_tool.v27545_smoke:main"
    )
    assert "setuptools" not in {dependency.lower() for dependency in project.dependencies}


def test_malicious_old_record_escape_is_rejected_before_mutation(
    tmp_path: Path,
) -> None:
    project = refresh.load_project(_minimal_project(tmp_path / "tool"))
    venv = tmp_path / "venv"
    site = venv / "lib/python3.12/site-packages"
    scripts = venv / "bin"
    dist = site / "onnx_splitpoint_tool-2.75.41.dist-info"
    dist.mkdir(parents=True)
    scripts.mkdir(parents=True)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: onnx-splitpoint-tool\n"
        "Version: 2.75.41\n\n",
        encoding="utf-8",
    )
    outside = tmp_path / "must-not-be-touched.txt"
    outside.write_text("sentinel\n", encoding="utf-8")
    escape = os.path.relpath(outside, site).replace(os.sep, "/")
    (dist / "RECORD").write_text(
        _record_text([(escape, "", "")]),
        encoding="utf-8",
    )

    with pytest.raises(refresh.RefreshError, match="escapes venv/tool roots"):
        refresh.discover_old_artifacts(
            project,
            (site,),
            scripts,
            venv,
        )

    assert outside.read_text(encoding="utf-8") == "sentinel\n"
    assert dist.is_dir()


def test_canonical_metadata_cannot_claim_an_unrelated_launcher(
    tmp_path: Path,
) -> None:
    project = refresh.load_project(_minimal_project(tmp_path / "tool"))
    venv = tmp_path / "venv"
    site = venv / "lib/python3.12/site-packages"
    scripts = venv / "bin"
    dist = site / "onnx_splitpoint_tool-2.75.41.dist-info"
    helper_dist = site / "preserved_helper-9.1.0.dist-info"
    dist.mkdir(parents=True)
    helper_dist.mkdir()
    scripts.mkdir(parents=True)
    helper_launcher = scripts / "helper-cli"
    helper_launcher.write_text("unrelated helper launcher\n", encoding="utf-8")
    helper_launcher.chmod(0o755)
    helper_metadata = (
        "Metadata-Version: 2.1\n"
        "Name: preserved-helper\n"
        "Version: 9.1.0\n\n"
    )
    (helper_dist / "METADATA").write_text(helper_metadata, encoding="utf-8")
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: onnx-splitpoint-tool\n"
        "Version: 2.75.41\n\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[console_scripts]\nhelper-cli = preserved_helper.cli:main\n",
        encoding="utf-8",
    )
    (dist / "RECORD").write_text(
        _record_text([(f"{dist.name}/RECORD", "", "")]),
        encoding="utf-8",
    )

    with pytest.raises(refresh.RefreshError, match="non-project console target"):
        refresh.discover_old_artifacts(project, (site,), scripts, venv)

    assert helper_launcher.read_text(encoding="utf-8") == (
        "unrelated helper launcher\n"
    )
    assert (helper_dist / "METADATA").read_text(encoding="utf-8") == (
        helper_metadata
    )
    assert dist.is_dir()


@pytest.mark.parametrize("control_name", ("METADATA", "entry_points.txt", "RECORD"))
def test_old_metadata_control_symlinks_are_rejected(
    tmp_path: Path,
    control_name: str,
) -> None:
    project = refresh.load_project(_minimal_project(tmp_path / "tool"))
    venv = tmp_path / "venv"
    site = venv / "lib/python3.12/site-packages"
    scripts = venv / "bin"
    dist = site / "onnx_splitpoint_tool-2.75.41.dist-info"
    dist.mkdir(parents=True)
    scripts.mkdir(parents=True)
    controls = {
        "METADATA": (
            "Metadata-Version: 2.1\n"
            "Name: onnx-splitpoint-tool\n"
            "Version: 2.75.41\n\n"
        ),
        "entry_points.txt": "[console_scripts]\n",
        "RECORD": f"{dist.name}/RECORD,,\n",
    }
    external = tmp_path / f"external-{control_name}"
    external.write_text(controls[control_name], encoding="utf-8")
    for name, payload in controls.items():
        path = dist / name
        if name == control_name:
            path.symlink_to(external)
        else:
            path.write_text(payload, encoding="utf-8")

    with pytest.raises(refresh.RefreshError, match="symlink"):
        refresh.discover_old_artifacts(project, (site,), scripts, venv)

    assert external.read_text(encoding="utf-8") == controls[control_name]
    assert dist.is_dir()


def test_transaction_rolls_back_old_metadata_pth_and_launcher(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = refresh.load_project(_minimal_project(tmp_path / "tool"))
    venv = tmp_path / "venv"
    site = venv / "lib/python3.12/site-packages"
    scripts = venv / "bin"
    dist = site / "onnx_splitpoint_tool-2.75.41.dist-info"
    dist.mkdir(parents=True)
    scripts.mkdir(parents=True)
    pth = site / "__editable__.onnx_splitpoint_tool-2.75.41.pth"
    launcher = scripts / "onnx-splitpoint-smoke-v27541"
    pth.write_text("/old/source\n", encoding="utf-8")
    old_entry_value = "onnx_splitpoint_tool.smoke:main"
    old_launcher_payload = refresh._launcher_content(old_entry_value)
    launcher.write_bytes(old_launcher_payload)
    launcher.chmod(0o755)
    (dist / "METADATA").write_text(
        "Metadata-Version: 2.1\n"
        "Name: onnx-splitpoint-tool\n"
        "Version: 2.75.41\n\n",
        encoding="utf-8",
    )
    (dist / "entry_points.txt").write_text(
        "[console_scripts]\n"
        f"onnx-splitpoint-smoke-v27541 = {old_entry_value}\n",
        encoding="utf-8",
    )
    (dist / "RECORD").write_text(
        _record_text(
            [
                (pth.name, "", ""),
                (os.path.relpath(launcher, site).replace(os.sep, "/"), "", ""),
                (f"{dist.name}/RECORD", "", ""),
            ]
        ),
        encoding="utf-8",
    )
    original_record = (dist / "RECORD").read_bytes()

    monkeypatch.setattr(
        refresh,
        "_site_and_script_paths",
        lambda: ((site,), scripts, venv),
    )

    def fail_verification(*_args: object, **_kwargs: object) -> None:
        raise refresh.RefreshError("injected verification failure")

    monkeypatch.setattr(refresh, "verify_install", fail_verification)
    with pytest.raises(refresh.RefreshError, match="injected verification failure"):
        refresh.refresh(
            project,
            expected_version="2.75.42",
            required={
                "onnx-splitpoint-smoke-v27542": (
                    "onnx_splitpoint_tool.smoke:main"
                ),
            },
            run_entry_points=(),
        )

    assert pth.read_text(encoding="utf-8") == "/old/source\n"
    assert launcher.read_bytes() == old_launcher_payload
    assert (dist / "RECORD").read_bytes() == original_record
    assert not list(venv.rglob("*.refresh-new-*"))
    assert not list(venv.rglob("*.refresh-backup-*"))
    assert not list(site.glob("onnx_splitpoint_tool-2.75.42.dist-info"))
