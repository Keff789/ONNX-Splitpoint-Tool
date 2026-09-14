from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import py_compile
import shutil
import subprocess
import sys

import pytest

from scripts import refresh_editable_install


ROOT = Path(__file__).resolve().parents[1]


def _prefix_cache_path(source: Path, prefix: Path) -> Path:
    previous = sys.pycache_prefix
    try:
        sys.pycache_prefix = str(prefix)
        return Path(importlib.util.cache_from_source(str(source.resolve())))
    finally:
        sys.pycache_prefix = previous


def _craft_timestamp_valid_malicious_cache(
    project: Path,
    *,
    marker: Path,
    attacker_prefix: Path,
) -> Path:
    """Leave benign source with a same-size/timestamp malicious cache."""

    source = project / "victim.py"
    template = (
        "import os\n"
        "from pathlib import Path\n\n"
        "def main():\n"
        f"    Path({str(marker)!r}).write_text({{expression}}, encoding='utf-8')\n"
        "    return 0\n"
    )
    malicious = template.format(expression="'MALICIOUS_BYTECODE_EXECUTED'")
    benign = template.format(
        expression="os.environ.get('PYTHONPYCACHEPREFIX', '')"
    )
    width = max(len(malicious), len(benign))
    malicious = malicious.ljust(width)
    benign = benign.ljust(width)
    source.write_text(malicious, encoding="utf-8")
    source_stat = source.stat()
    local_cache = Path(importlib.util.cache_from_source(str(source)))
    py_compile.compile(str(source), cfile=str(local_cache), doraise=True)

    attacker_cache = _prefix_cache_path(source, attacker_prefix)
    attacker_cache.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(local_cache, attacker_cache)
    source.write_text(benign, encoding="utf-8")
    os.utime(
        source,
        ns=(source_stat.st_atime_ns, source_stat.st_mtime_ns),
    )
    assert source.stat().st_size == source_stat.st_size
    return source


def _attacker_environment(project: Path, attacker_prefix: Path) -> dict[str, str]:
    return {
        **os.environ,
        "PYTHONPATH": str(project),
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONPYCACHEPREFIX": str(attacker_prefix),
    }


def _prove_cache_is_timestamp_valid(
    project: Path,
    attacker_prefix: Path,
    marker: Path,
) -> None:
    vulnerable = subprocess.run(
        [sys.executable, "-B", "-c", "import victim; raise SystemExit(victim.main())"],
        cwd=project,
        env=_attacker_environment(project, attacker_prefix),
        text=True,
        capture_output=True,
        check=False,
    )
    assert vulnerable.returncode == 0, vulnerable.stdout + vulnerable.stderr
    assert marker.read_text(encoding="utf-8") == "MALICIOUS_BYTECODE_EXECUTED"
    marker.unlink()


def test_generated_console_launcher_ignores_source_and_inherited_bytecode_cache(
    tmp_path: Path,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    marker = tmp_path / "launcher-marker.txt"
    attacker_prefix = tmp_path / "attacker-cache"
    _craft_timestamp_valid_malicious_cache(
        project,
        marker=marker,
        attacker_prefix=attacker_prefix,
    )
    _prove_cache_is_timestamp_valid(project, attacker_prefix, marker)

    launcher = tmp_path / "safe-launcher"
    launcher.write_bytes(refresh_editable_install._launcher_content("victim:main"))
    launcher.chmod(0o755)
    shell_check = subprocess.run(
        ["/bin/sh", "-n", str(launcher)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert shell_check.returncode == 0, shell_check.stdout + shell_check.stderr
    completed = subprocess.run(
        [str(launcher)],
        cwd=project,
        env=_attacker_environment(project, attacker_prefix),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed_prefix = marker.read_text(encoding="utf-8")
    assert observed_prefix.startswith(
        "/tmp/onnx-splitpoint-console-pycache."
    )
    assert observed_prefix != str(attacker_prefix)
    assert not Path(observed_prefix).exists()


def test_start_gui_bootstrap_ignores_source_and_inherited_bytecode_cache(
    tmp_path: Path,
) -> None:
    project = tmp_path / "gui-project"
    project.mkdir()
    marker = tmp_path / "gui-marker.txt"
    attacker_prefix = tmp_path / "attacker-cache"
    _craft_timestamp_valid_malicious_cache(
        project,
        marker=marker,
        attacker_prefix=attacker_prefix,
    )
    _prove_cache_is_timestamp_valid(project, attacker_prefix, marker)

    shutil.copy2(ROOT / "start_gui.sh", project / "start_gui.sh")
    (project / "start_gui.sh").chmod(0o755)
    bin_dir = project / ".venv" / "bin"
    bin_dir.mkdir(parents=True)
    (bin_dir / "python").symlink_to(Path(sys.executable).resolve())
    (bin_dir / "activate").write_text(
        "deactivate() { :; }\n",
        encoding="utf-8",
    )
    (project / ".venv" / ".osp_gui_core_deps_ok").touch()
    (project / "analyse_and_split_gui.py").write_text(
        "import victim\nraise SystemExit(victim.main())\n",
        encoding="utf-8",
    )

    completed = subprocess.run(
        ["bash", "start_gui.sh"],
        cwd=project,
        env=_attacker_environment(project, attacker_prefix),
        text=True,
        capture_output=True,
        check=False,
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    observed_prefix = marker.read_text(encoding="utf-8")
    assert observed_prefix.startswith("/tmp/onnx-splitpoint-gui-pycache.")
    assert observed_prefix != str(attacker_prefix)
    assert not Path(observed_prefix).exists()


@pytest.mark.parametrize(
    "relative",
    [
        "scripts/update_source_release.sh",
        "scripts/run_v27912_small_acceptance.sh",
        "scripts/run_v27912_seven_model_long_overnight.sh",
        "scripts/run_v27913_small_acceptance.sh",
        "scripts/run_v27913_seven_model_long_overnight.sh",
    ],
)
def test_release_shell_entrypoints_override_bytecode_environment_before_python(
    relative: str,
) -> None:
    source = (ROOT / relative).read_text(encoding="utf-8")
    isolation = source.index("export PYTHONPYCACHEPREFIX=")
    assert "/usr/bin/mktemp -d /tmp/" in source
    first_python = min(
        position
        for token in ('"$PYTHON"', '"$TOOL_PYTHON"')
        if (position := source.find(token)) >= 0
    )
    assert source.index("export PYTHONDONTWRITEBYTECODE=1") < first_python
    assert isolation < first_python
