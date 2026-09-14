from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from scripts import build_source_manifest
from onnx_splitpoint_tool.release_identity import BUILD_ID, VERSION
from onnx_splitpoint_tool.source_integrity import (
    SOURCE_MANIFEST_SCHEMA,
    create_source_integrity_binding,
    verify_installed_source_integrity,
    verify_source_integrity_binding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _release_tree(tmp_path: Path) -> Path:
    root = tmp_path / "release"
    files = {
        "pyproject.toml": f'[project]\nname = "test"\nversion = "{VERSION}"\n',
        "onnx_splitpoint_tool/release_identity.py": (
            f'VERSION = "{VERSION}"\nBUILD_ID = "{BUILD_ID}"\n'
        ),
        # Keep one real one-byte row to exercise Python's bool==1 edge case.
        "onnx_splitpoint_tool/runtime.py": "x",
        "scripts/run.py": "raise SystemExit(0)\n",
        "profiles/shipped.yaml": "name: shipped\n",
    }
    for relative, content in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    rows = [
        {
            "path": path.relative_to(root).as_posix(),
            "size": int(path.stat().st_size),
            "sha256": _sha256(path),
        }
        for path in sorted(
            (root / relative for relative in files),
            key=lambda candidate: candidate.relative_to(root).as_posix(),
        )
    ]
    manifest = {
        "schema": SOURCE_MANIFEST_SCHEMA,
        "package_version": VERSION,
        "workflow_version": BUILD_ID,
        "file_count": len(rows),
        "files": rows,
    }
    (root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in rows),
        encoding="utf-8",
    )
    return root


def test_installed_source_integrity_allows_only_preserved_top_level_profiles(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    custom = root / "profiles" / "user custom.yml"
    custom.write_text("name: user\n", encoding="utf-8")

    result = verify_installed_source_integrity(root)

    assert result["ok"] is True
    assert result["status"] == "verified"
    assert result["manifest_sha256"] == _sha256(root / "SOURCE_MANIFEST.json")
    assert [row["path"] for row in result["user_profiles"]] == [
        "profiles/user custom.yml"
    ]
    assert result["unexpected_extras"] == []
    assert result["symlinks"] == []


def test_installed_source_integrity_detects_mutated_release_file(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    (root / "onnx_splitpoint_tool" / "runtime.py").write_text(
        "VALUE = 2\n", encoding="utf-8"
    )

    result = verify_installed_source_integrity(root)

    assert result["ok"] is False
    assert result["checks"]["rows_ok"] is False
    assert [row["path"] for row in result["changed"]] == [
        "onnx_splitpoint_tool/runtime.py"
    ]
    assert "release_files_changed" in result["errors"]


def test_installed_source_integrity_rejects_unmanifested_allowlisted_file(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    extra = root / "scripts" / "unexpected.py"
    extra.write_text("print('unexpected')\n", encoding="utf-8")

    result = verify_installed_source_integrity(root)

    assert result["ok"] is False
    assert result["checks"]["unexpected_extras_ok"] is False
    assert [row["path"] for row in result["unexpected_extras"]] == [
        "scripts/unexpected.py"
    ]
    assert "unexpected_allowlisted_files" in result["errors"]


@pytest.mark.parametrize(
    "relative_path",
    (
        "sitecustomize.py",
        "sitecustomize.pyc",
        "unexpected.log",
        "scripts/sourceless.pyc",
        "unknown/payload.txt",
        "tmp/payload.py",
        ".idea/sitecustomize.py",
        ".git",
    ),
)
def test_all_verifiers_reject_unknown_regular_files_regardless_of_suffix(
    tmp_path: Path,
    relative_path: str,
) -> None:
    root = _release_tree(tmp_path)
    extra = root / relative_path
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_bytes(b"unmanifested executable input")

    with pytest.raises(RuntimeError, match="unexpected source entries"):
        build_source_manifest.build(root)

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    for result in (release, installed, runtime):
        assert result["ok"] is False
        assert result["checks"]["unexpected_extras_ok"] is False
        assert relative_path in {
            str(row["path"]) for row in result["unexpected_extras"]
        }


def test_all_verifiers_reject_unknown_top_level_symlink(tmp_path: Path) -> None:
    root = _release_tree(tmp_path)
    link = root / "sitecustomize.py"
    link.symlink_to(root / "pyproject.toml")

    with pytest.raises(RuntimeError, match="source/index symlinks"):
        build_source_manifest.build(root)

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    for result in (release, installed, runtime):
        assert result["ok"] is False
        assert "sitecustomize.py" in result["symlinks"]


def test_all_verifiers_reject_symlinked_source_root(tmp_path: Path) -> None:
    real_root = _release_tree(tmp_path)
    root = tmp_path / "release-link"
    root.symlink_to(real_root, target_is_directory=True)

    with pytest.raises(RuntimeError, match="source/index symlinks"):
        build_source_manifest.build(root)

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    assert release["ok"] is False
    assert installed["ok"] is False
    assert release["symlinks"] == ["."]
    assert installed["symlinks"] == ["."]
    assert runtime["ok"] is False
    assert runtime["status"] == "source_root_symlink"
    assert runtime["symlinks"] == ["."]


@pytest.mark.parametrize("index_name", ("SOURCE_MANIFEST.json", "SHA256SUMS.txt"))
def test_verifiers_reject_non_regular_index_before_reading(
    tmp_path: Path,
    index_name: str,
) -> None:
    root = _release_tree(tmp_path)
    index = root / index_name
    index.rename(root / f"{index_name}.saved")
    index.mkdir()

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    assert release["ok"] is False
    assert installed["ok"] is False
    assert release["checks"]["source_indexes_regular"] is False
    assert installed["checks"]["source_indexes_regular"] is False
    assert release["invalid_indexes"] == [index_name]
    assert installed["invalid_indexes"] == [index_name]
    assert runtime["ok"] is False
    assert runtime["status"] == "source_indexes_unreadable"


def test_real_explicit_cache_directory_is_pruned_not_name_based_file(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    cache = root / "__pycache__"
    cache.mkdir()
    (cache / "sitecustomize.cpython-312.pyc").write_bytes(b"cache")

    assert build_source_manifest.verify(root, scope="release")["ok"] is True
    assert build_source_manifest.verify(root, scope="installed")["ok"] is True
    assert verify_installed_source_integrity(root)["ok"] is True

    # The same executable-looking suffix at top level is not hidden merely by
    # EXCLUDED_FILE_SUFFIXES.
    (root / "sitecustomize.pyc").write_bytes(b"sourceless")
    assert build_source_manifest.verify(root, scope="release")["ok"] is False
    assert verify_installed_source_integrity(root)["ok"] is False


@pytest.mark.parametrize("target", ["source", "index"])
def test_installed_source_integrity_rejects_source_and_index_symlinks(
    tmp_path: Path,
    target: str,
) -> None:
    root = _release_tree(tmp_path)
    if target == "source":
        link = root / "onnx_splitpoint_tool" / "linked.py"
        link.symlink_to(root / "onnx_splitpoint_tool" / "runtime.py")
        expected = "onnx_splitpoint_tool/linked.py"
    else:
        sums = root / "SHA256SUMS.txt"
        saved = root / "sums.saved"
        sums.rename(saved)
        sums.symlink_to(saved)
        expected = "SHA256SUMS.txt"

    result = verify_installed_source_integrity(root)

    assert result["ok"] is False
    assert expected in result["symlinks"]
    assert result["status"] in {
        "source_index_symlink",
        "source_integrity_failed",
    }


@pytest.mark.parametrize(
    "directory_name",
    ["nested", ".git", "__pycache__", "tmp", "plugin.egg-info"],
)
def test_installed_source_integrity_rejects_every_profile_subdirectory(
    tmp_path: Path,
    directory_name: str,
) -> None:
    root = _release_tree(tmp_path)
    nested = root / "profiles" / directory_name
    nested.mkdir()
    (nested / "user.yaml").write_text("name: nested\n", encoding="utf-8")

    result = verify_installed_source_integrity(root)

    assert result["ok"] is False
    assert result["unexpected_extras"] == [
        {"path": f"profiles/{directory_name}", "kind": "directory"}
    ]


def test_source_integrity_binding_reverifies_exact_authoritative_indexes(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    original = verify_installed_source_integrity(root)
    binding = create_source_integrity_binding(original)

    verified = verify_source_integrity_binding(binding, root=root)
    assert verified["ok"] is True
    assert verified["manifest_sha256"] == original["manifest_sha256"]

    changed = dict(binding)
    changed["manifest_sha256"] = "f" * 64
    rejected = verify_source_integrity_binding(changed, root=root)
    assert rejected["ok"] is False
    assert "current_source_manifest_sha256_mismatch" in rejected["errors"]

    (root / "onnx_splitpoint_tool" / "runtime.py").write_text(
        "VALUE = 99\n", encoding="utf-8"
    )
    mutated = verify_source_integrity_binding(binding, root=root)
    assert mutated["ok"] is False
    assert "current_installed_source_integrity_failed" in mutated["errors"]


def test_source_integrity_binding_rejects_float_schema_version(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    report = verify_installed_source_integrity(root)
    binding = create_source_integrity_binding(report)
    binding["schema_version"] = 1.0

    result = verify_source_integrity_binding(binding, root=root)

    assert result["ok"] is False
    assert "source_integrity_binding_version_invalid" in result["errors"]


@pytest.mark.parametrize(
    "mutation",
    ["extra_top_level", "row_size_bool", "row_size_float", "file_count_float"],
)
def test_build_and_runtime_manifest_shape_reject_the_same_malformed_contract(
    tmp_path: Path,
    mutation: str,
) -> None:
    root = _release_tree(tmp_path)
    manifest_path = root / "SOURCE_MANIFEST.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if mutation == "extra_top_level":
        payload["unexpected"] = "must-fail"
    elif mutation == "file_count_float":
        payload["file_count"] = float(payload["file_count"])
    else:
        row = next(
            item
            for item in payload["files"]
            if item["path"] == "onnx_splitpoint_tool/runtime.py"
        )
        assert row["size"] == 1
        row["size"] = True if mutation == "row_size_bool" else 1.0
    manifest_path.write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    assert release["ok"] is False
    assert release["checks"]["manifest_shape_ok"] is False
    assert installed["ok"] is False
    assert installed["checks"]["manifest_shape_ok"] is False
    assert runtime["ok"] is False
    assert runtime["checks"]["manifest_shape_ok"] is False


def test_build_and_runtime_reject_crlf_sha256sums_projection(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    sums = root / "SHA256SUMS.txt"
    sums.write_bytes(sums.read_bytes().replace(b"\n", b"\r\n"))

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    assert release["checks"]["sha256sums_ok"] is False
    assert installed["checks"]["sha256sums_ok"] is False
    assert runtime["checks"]["sha256sums_ok"] is False


def test_build_release_and_runtime_reject_literal_backslash_source_name(
    tmp_path: Path,
) -> None:
    root = _release_tree(tmp_path)
    invalid = root / "scripts" / "invalid\\name.py"
    invalid.write_text("raise SystemExit(0)\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="invalid source manifest paths"):
        build_source_manifest.build(root)

    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    assert release["ok"] is False
    assert release["checks"]["manifest_paths_ok"] is False
    assert release["invalid_paths"] == ["scripts/invalid\\name.py"]
    assert installed["ok"] is False
    assert runtime["ok"] is False


@pytest.mark.parametrize("target", ["source", "index", "allowed_root"])
def test_build_release_and_runtime_never_admit_or_hash_source_index_symlink(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    target: str,
) -> None:
    root = _release_tree(tmp_path)
    if target == "allowed_root":
        link = root / "start_gui.sh"
        link.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        build_source_manifest.build(root)
        saved = root / "start_gui.sh.saved"
    elif target == "source":
        link = root / "scripts" / "run.py"
        saved = root / "run.py.saved"
    else:
        link = root / "SHA256SUMS.txt"
        saved = root / "sums.saved"
    link.rename(saved)
    link.symlink_to(saved)
    logical = link.relative_to(root).as_posix()

    hash_calls: list[str] = []
    original_hash = build_source_manifest._sha256

    def guarded_hash(path: Path) -> str:
        assert not path.is_symlink(), "release verifier followed a symlink"
        hash_calls.append(path.relative_to(root).as_posix())
        return original_hash(path)

    monkeypatch.setattr(build_source_manifest, "_sha256", guarded_hash)

    with pytest.raises(RuntimeError, match="source/index symlinks"):
        build_source_manifest.build(root)
    release = build_source_manifest.verify(root, scope="release")
    installed = build_source_manifest.verify(root, scope="installed")
    runtime = verify_installed_source_integrity(root)

    assert hash_calls == []
    assert release["ok"] is False
    assert release["checks"]["symlinks_ok"] is False
    assert release["symlinks"] == [logical]
    assert installed["ok"] is False
    assert installed["checks"]["symlinks_ok"] is False
    assert runtime["ok"] is False
    assert logical in runtime["symlinks"]
