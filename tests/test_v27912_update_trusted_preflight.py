from __future__ import annotations

import hashlib
import importlib.util
import json
import marshal
import os
from pathlib import Path
import stat
import struct
import subprocess
import sys
import tempfile
import unittest
import zipfile


ROOT = Path(__file__).resolve().parents[1]
UPDATER = ROOT / "scripts" / "update_source_release.sh"
VERSION = "2.79.13"
BUILD_ID = "v2.79.13-platform-power-calibration-operational-repair"
PREFIX = f"ONNX-Splitpoint-Tool_v{VERSION}"


def _member(name: str, *, mode: int = stat.S_IFREG | 0o644) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.create_system = 3
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = mode << 16
    return info


def _timestamp_pyc(source: bytes, filename: str) -> bytes:
    code = compile(source, filename, "exec")
    return b"".join(
        (
            importlib.util.MAGIC_NUMBER,
            struct.pack("<I", 0),
            struct.pack("<II", 0, len(source)),
            marshal.dumps(code),
        )
    )


def _write_release(
    path: Path,
    *,
    injected_name: str | None,
    injected_mode: int,
    execution_marker: Path,
) -> None:
    files = {
        "onnx_splitpoint_tool/release_identity.py": (
            f'VERSION = "{VERSION}"\nBUILD_ID = "{BUILD_ID}"\n'
        ).encode(),
        "pyproject.toml": (
            f'[project]\nname = "onnx-splitpoint-tool"\nversion = "{VERSION}"\n'
        ).encode(),
        "scripts/run.py": b"raise SystemExit(0)\n",
    }
    rows = [
        {
            "path": name,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        for name, payload in sorted(files.items())
    ]
    manifest = (
        json.dumps(
            {
                "schema": "onnx-splitpoint/source-manifest-v1",
                "package_version": VERSION,
                "workflow_version": BUILD_ID,
                "file_count": len(rows),
                "files": rows,
            },
            indent=2,
        )
        + "\n"
    ).encode()
    sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in rows
    ).encode()
    members = {
        **files,
        "SOURCE_MANIFEST.json": manifest,
        "SHA256SUMS.txt": sums,
    }
    with zipfile.ZipFile(path, "w") as archive:
        for name, payload in sorted(members.items()):
            archive.writestr(_member(f"{PREFIX}/{name}"), payload)
        if injected_name is not None:
            injected_source = (
                f"open({str(execution_marker)!r}, 'w').write('executed')\n"
            ).encode()
            injected_payload = (
                _timestamp_pyc(injected_source, injected_name)
                if injected_name.endswith(".pyc")
                else injected_source
            )
            if stat.S_ISLNK(injected_mode):
                injected_payload = b"pyproject.toml"
            archive.writestr(
                _member(f"{PREFIX}/{injected_name}", mode=injected_mode),
                injected_payload,
            )


def _tool_fixture(root: Path) -> tuple[Path, Path, dict[str, str]]:
    home = root / "home"
    tool = root / "tool"
    for directory in (
        tool / "onnx_splitpoint_tool",
        tool / "scripts",
        tool / ".venv" / "bin",
        home,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    for relative in (
        "pyproject.toml",
        "SOURCE_MANIFEST.json",
        "onnx_splitpoint_tool/__init__.py",
        "scripts/build_source_manifest.py",
    ):
        (tool / relative).write_text("fixture\n", encoding="utf-8")
    (tool / ".venv" / "bin" / "python").symlink_to(
        Path(sys.executable).resolve()
    )

    extraction_marker = root / "unzip-called"
    probe_bin = root / "probe-bin"
    probe_bin.mkdir()
    unzip_probe = probe_bin / "unzip"
    unzip_probe.write_text(
        "#!/bin/bash\n"
        "set -eu\n"
        ': > "${EXTRACTION_MARKER:?}"\n'
        "exit 99\n",
        encoding="utf-8",
    )
    unzip_probe.chmod(0o755)

    import_marker = root / "sitecustomize-executed"
    hostile_pythonpath = root / "hostile-pythonpath"
    hostile_pythonpath.mkdir()
    (hostile_pythonpath / "sitecustomize.py").write_text(
        f"open({str(import_marker)!r}, 'w').write('executed')\n",
        encoding="utf-8",
    )
    env = {
        **os.environ,
        "HOME": str(home),
        "PATH": f"{probe_bin}:{os.environ['PATH']}",
        "PYTHONPATH": str(hostile_pythonpath),
        "EXTRACTION_MARKER": str(extraction_marker),
    }
    return tool, extraction_marker, env


class TrustedUpdaterPreflightTests(unittest.TestCase):
    def test_valid_archive_reaches_extractor_after_trusted_preflight(self) -> None:
        with tempfile.TemporaryDirectory() as raw:
            root = Path(raw)
            archive = root / "release.zip"
            archive_execution_marker = root / "archive-payload-executed"
            _write_release(
                archive,
                injected_name=None,
                injected_mode=stat.S_IFREG | 0o644,
                execution_marker=archive_execution_marker,
            )
            tool, extraction_marker, env = _tool_fixture(root)
            result = subprocess.run(
                ["bash", str(UPDATER), str(archive), str(tool)],
                check=False,
                capture_output=True,
                text=True,
                env=env,
            )
            # The fake extractor deliberately returns 99.  Reaching it proves
            # the valid archive crossed the trusted admission boundary.
            self.assertEqual(result.returncode, 99, result.stderr)
            self.assertTrue(extraction_marker.exists())
            self.assertFalse((root / "sitecustomize-executed").exists())

    def test_untrusted_members_are_rejected_before_extraction_or_import(self) -> None:
        cases = {
            "sitecustomize.py": stat.S_IFREG | 0o644,
            "sitecustomize.pyc": stat.S_IFREG | 0o644,
            "scripts/hashlib.py": stat.S_IFREG | 0o644,
            "onnx_splitpoint_tool/link.py": stat.S_IFLNK | 0o777,
        }
        for injected_name, injected_mode in cases.items():
            with self.subTest(injected_name=injected_name), tempfile.TemporaryDirectory() as raw:
                root = Path(raw)
                archive = root / "release.zip"
                archive_execution_marker = root / "archive-payload-executed"
                _write_release(
                    archive,
                    injected_name=injected_name,
                    injected_mode=injected_mode,
                    execution_marker=archive_execution_marker,
                )
                tool, extraction_marker, env = _tool_fixture(root)
                result = subprocess.run(
                    ["bash", str(UPDATER), str(archive), str(tool)],
                    check=False,
                    capture_output=True,
                    text=True,
                    env=env,
                )
                self.assertEqual(result.returncode, 65, result.stderr)
                self.assertIn("vertrauenswürdige Release-Prüfung", result.stderr)
                self.assertFalse(extraction_marker.exists())
                self.assertFalse((root / "sitecustomize-executed").exists())
                self.assertFalse(archive_execution_marker.exists())

    def test_trusted_checks_precede_candidate_execution(self) -> None:
        source = UPDATER.read_text(encoding="utf-8")
        archive_check = source.index("trusted_source_preflight archive")
        extraction = source.index('unzip -q "$ZIP"')
        tree_check = source.index("trusted_source_preflight tree", archive_check)
        staged_script = source.index('"$SOURCE_DIR/scripts/build_source_manifest.py"')
        self.assertLess(archive_check, extraction)
        self.assertLess(extraction, tree_check)
        self.assertLess(tree_check, staged_script)
        self.assertIn(
            'PYTHONPATH= PYTHONHOME= "$PYTHON" -I -S -B - "$mode"',
            source,
        )


if __name__ == "__main__":
    unittest.main()
