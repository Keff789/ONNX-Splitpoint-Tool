from __future__ import annotations

import csv
from pathlib import Path

import pytest

from scripts import refresh_editable_install as refresh


def test_metadata_restore_collision_preserves_error_backup_and_other_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "tool"
    package = root / "onnx_splitpoint_tool"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "smoke.py").write_text("def main():\n    return 0\n")
    (root / "pyproject.toml").write_text(
        '[project]\nname = "onnx-splitpoint-tool"\nversion = "2.79.21"\n'
        '[project.scripts]\nonnx-splitpoint-smoke = "onnx_splitpoint_tool.smoke:main"\n'
    )
    project = refresh.load_project(root)
    venv = tmp_path / "venv"
    site = venv / "lib/python3.12/site-packages"
    scripts = venv / "bin"
    dist = site / "onnx_splitpoint_tool-2.79.20.dist-info"
    dist.mkdir(parents=True)
    scripts.mkdir(parents=True)
    old_metadata = "Metadata-Version: 2.1\nName: onnx-splitpoint-tool\nVersion: 2.79.20\n"
    (dist / "METADATA").write_text(old_metadata)
    entry_value = "onnx_splitpoint_tool.smoke:main"
    (dist / "entry_points.txt").write_text(
        f"[console_scripts]\nonnx-splitpoint-smoke = {entry_value}\n"
    )
    launcher = scripts / "onnx-splitpoint-smoke"
    old_launcher = refresh._launcher_content(entry_value)
    launcher.write_bytes(old_launcher)
    launcher.chmod(0o755)
    pth = site / "__editable__.onnx_splitpoint_tool-2.79.20.pth"
    pth.write_text("/prior/source\n")
    with (dist / "RECORD").open("w", newline="") as handle:
        csv.writer(handle).writerows([
            (pth.name, "", ""),
            (refresh._record_name(launcher, site), "", ""),
            (f"{dist.name}/RECORD", "", ""),
        ])
    monkeypatch.setattr(refresh, "_site_and_script_paths", lambda: ((site,), scripts, venv))

    def failed_verification(*_args):
        # Reproduce the observed failure: another writer recreated the old
        # metadata path between its backup rename and the rollback.
        dist.mkdir()
        (dist / "concurrent.txt").write_text("preserve concurrent writer\n")
        raise refresh.RefreshError("original fresh-process verification failed")

    monkeypatch.setattr(refresh, "verify_install", failed_verification)
    with pytest.raises(refresh.RefreshError, match="original fresh-process verification failed") as failure:
        refresh.refresh(
            project, expected_version="2.79.21",
            required={"onnx-splitpoint-smoke": entry_value}, run_entry_points=(),
        )
    assert "rollback incomplete" in str(failure.value)
    assert "rollback destination reappeared" in str(failure.value)
    assert isinstance(failure.value.__cause__, refresh.RefreshError)
    assert (dist / "concurrent.txt").read_text() == "preserve concurrent writer\n"
    backups = list(site.glob(".onnx_splitpoint_tool-2.79.20.dist-info.refresh-backup-*"))
    assert len(backups) == 1
    assert (backups[0] / "METADATA").read_text() == old_metadata
    assert launcher.read_bytes() == old_launcher
    assert pth.read_text() == "/prior/source\n"
    assert not (site / "onnx_splitpoint_tool-2.79.21.dist-info").exists()
    assert not list(venv.rglob("*.refresh-new-*"))
