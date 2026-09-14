from __future__ import annotations

import fcntl
import os
from pathlib import Path
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
UPDATER = ROOT / "scripts/update_source_release.sh"
LONGRUN_LAUNCHER = ROOT / "scripts/run_v27910_seven_model_long_overnight.sh"


def _fixture(tmp_path: Path) -> tuple[Path, Path, dict[str, str], Path]:
    home = tmp_path / "home"
    tool = tmp_path / "tool"
    lock_dir = home / ".onnx_splitpoint_tool" / "locks"
    for relative in (
        "onnx_splitpoint_tool",
        "scripts",
        ".venv/bin",
    ):
        (tool / relative).mkdir(parents=True, exist_ok=True)
    for relative in (
        "pyproject.toml",
        "SOURCE_MANIFEST.json",
        "onnx_splitpoint_tool/__init__.py",
        "scripts/build_source_manifest.py",
    ):
        (tool / relative).write_text("\n", encoding="utf-8")
    (tool / ".venv/bin/python").symlink_to(Path(sys.executable).resolve())
    lock_dir.mkdir(parents=True)
    release = tmp_path / "invalid-release.zip"
    release.write_bytes(b"not a zip; the lock gate must run before unzip")
    env = {**os.environ, "HOME": str(home)}
    return tool, release, env, lock_dir


def _run(release: Path, tool: Path, env: dict[str, str], *extra: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["bash", str(UPDATER), str(release), str(tool), *extra],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )


def _wait_for_file(path: Path, process: subprocess.Popen[str]) -> None:
    deadline = time.monotonic() + 10.0
    while time.monotonic() < deadline:
        if path.exists():
            return
        if process.poll() is not None:
            stdout, stderr = process.communicate()
            raise AssertionError(
                f"process exited before synchronization: rc={process.returncode}\n"
                f"stdout={stdout}\nstderr={stderr}"
            )
        time.sleep(0.02)
    raise AssertionError(f"timed out waiting for {path}")


def _install_unzip_probe(
    tmp_path: Path,
    env: dict[str, str],
    *,
    blocking: bool,
) -> tuple[Path, Path | None]:
    bin_dir = tmp_path / "probe-bin"
    bin_dir.mkdir(exist_ok=True)
    marker = tmp_path / "unzip-called"
    release = tmp_path / "release-unzip" if blocking else None
    body = [
        "#!/bin/bash",
        "set -eu",
        ': > "${EXTRACTION_MARKER:?}"',
    ]
    if blocking:
        body.extend(
            (
                'while [[ ! -e "${EXTRACTION_RELEASE:?}" ]]; do',
                "  /bin/sleep 0.02",
                "done",
            )
        )
    body.append("exit 65")
    probe = bin_dir / "unzip"
    probe.write_text("\n".join(body) + "\n", encoding="utf-8")
    probe.chmod(0o755)
    env["PATH"] = f"{bin_dir}:{env['PATH']}"
    env["EXTRACTION_MARKER"] = str(marker)
    if release is not None:
        env["EXTRACTION_RELEASE"] = str(release)
    return marker, release


def test_held_workflow_lock_blocks_before_archive_extraction(tmp_path: Path) -> None:
    tool, release, env, lock_dir = _fixture(tmp_path)
    lock = lock_dir / "active-campaign.lock"
    with lock.open("w+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(release, tool, env)
    assert result.returncode == 75
    assert "Quellupdate blockiert" in result.stderr
    assert str(lock) in result.stderr
    assert "not a zip" not in result.stderr


def test_stale_unheld_lock_file_does_not_block(tmp_path: Path) -> None:
    tool, release, env, lock_dir = _fixture(tmp_path)
    (lock_dir / "stale-campaign.lock").write_text("stale\n", encoding="utf-8")
    result = _run(release, tool, env)
    assert result.returncode != 75
    assert "Quellupdate blockiert" not in result.stderr


def test_maintainer_override_is_explicit_and_noisy(tmp_path: Path) -> None:
    tool, release, env, lock_dir = _fixture(tmp_path)
    lock = lock_dir / "maintainer-recovery.lock"
    with lock.open("w+", encoding="utf-8") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        result = _run(
            release,
            tool,
            env,
            "--maintainer-allow-active-workflow",
        )
    assert result.returncode != 75
    assert "Maintainer-Override aktiv" in result.stderr
    assert str(lock) in result.stderr


def test_running_launcher_shared_interlock_blocks_updater_even_with_override(
    tmp_path: Path,
) -> None:
    tool, release_zip, env, _lock_dir = _fixture(tmp_path)
    extraction_marker, _ = _install_unzip_probe(
        tmp_path,
        env,
        blocking=False,
    )

    # Stop the real launcher immediately after it has acquired the shared
    # interlock but before its semantic venv check can finish.  The updater is
    # then raced against that real launcher process, not a stale lock filename.
    python_path = tool / ".venv/bin/python"
    python_path.unlink()
    launcher_entered = tmp_path / "launcher-entered-python"
    launcher_release = tmp_path / "release-launcher-python"
    python_path.write_text(
        "\n".join(
            (
                "#!/bin/bash",
                "set -eu",
                ': > "${LAUNCHER_ENTERED:?}"',
                'while [[ ! -e "${LAUNCHER_RELEASE:?}" ]]; do',
                "  /bin/sleep 0.02",
                "done",
                "exit 69",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    race_env = {
        **env,
        "TOOL": str(tool),
        "LAUNCHER_ENTERED": str(launcher_entered),
        "LAUNCHER_RELEASE": str(launcher_release),
    }
    launcher = subprocess.Popen(
        [
            "bash",
            str(LONGRUN_LAUNCHER),
            "--yolo11-r8b-output",
            "/tmp",
            "--yolov7-claim-output",
            "/tmp",
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=race_env,
    )
    try:
        _wait_for_file(launcher_entered, launcher)
        result = _run(
            release_zip,
            tool,
            race_env,
            "--maintainer-allow-active-workflow",
        )
        assert result.returncode == 75
        assert "Workflow/Plattform-Interlock" in result.stderr
        assert "nicht umgehen" in result.stderr
        assert not extraction_marker.exists()
    finally:
        launcher_release.touch()
        try:
            launcher.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            launcher.kill()
            launcher.communicate()


def test_running_updater_exclusive_interlock_blocks_new_longrun_launcher(
    tmp_path: Path,
) -> None:
    tool, release_zip, env, _lock_dir = _fixture(tmp_path)
    preflight_marker = tmp_path / "updater-entered-preflight"
    preflight_release = tmp_path / "release-updater-preflight"
    python_path = tool / ".venv/bin/python"
    python_path.unlink()
    python_path.write_text(
        "\n".join(
            (
                "#!/bin/bash",
                "set -eu",
                ': > "${UPDATER_PREFLIGHT_ENTERED:?}"',
                'while [[ ! -e "${UPDATER_PREFLIGHT_RELEASE:?}" ]]; do',
                "  /bin/sleep 0.02",
                "done",
                "exit 65",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    python_path.chmod(0o755)
    env = {
        **env,
        "UPDATER_PREFLIGHT_ENTERED": str(preflight_marker),
        "UPDATER_PREFLIGHT_RELEASE": str(preflight_release),
    }
    updater = subprocess.Popen(
        ["bash", str(UPDATER), str(release_zip), str(tool)],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
    )
    try:
        # The trusted archive preflight now deliberately precedes extraction.
        # Reaching its isolated interpreter proves that the updater already
        # owns its lifetime-long exclusive interlock.
        _wait_for_file(preflight_marker, updater)
        result = subprocess.run(
            [
                "bash",
                str(LONGRUN_LAUNCHER),
                "--yolo11-r8b-output",
                "/tmp",
                "--yolov7-claim-output",
                "/tmp",
            ],
            check=False,
            capture_output=True,
            text=True,
            env=env,
        )
        assert result.returncode == 75
        assert "exklusiven Workflow/Plattform-Interlock" in result.stderr
        assert "V279_TOOL_PYTHON_VENV=PASS" not in result.stdout
    finally:
        preflight_release.touch()
        try:
            updater.communicate(timeout=5)
        except subprocess.TimeoutExpired:
            updater.kill()
            updater.communicate()
