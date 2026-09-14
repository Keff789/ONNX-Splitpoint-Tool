#!/usr/bin/env python3
"""Build or verify the deterministic source-release hash indexes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import stat
from pathlib import Path, PurePosixPath
from typing import Any


EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".hypothesis",
    ".nox",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
}
EXCLUDED_FILE_NAMES = {".git", "SOURCE_MANIFEST.json", "SHA256SUMS.txt"}
EXCLUDED_CASEFOLD_FILE_NAMES = {
    ".coverage",
    ".directory",
    ".ds_store",
    "coverage.json",
    "coverage.xml",
    "desktop.ini",
    "thumbs.db",
}
EXCLUDED_FILE_PREFIXES = (".#", "._", ".coverage.")
ALLOWED_BINARY_TEST_FIXTURES = {
    'tests/fixtures/v282_h10_generic/models/yolo26m/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log',
    'tests/fixtures/v282_h10_generic/models/yolo26s/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log',

    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/repeat_retry_attempts/repeat_000/attempt_01/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_001/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_002/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/repeat_retry_attempts/repeat_000/attempt_01/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_001/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_002/workload_stdout.log',

    "tests/fixtures/v27930/semantic_probe_outputs.npz",
    "tests/fixtures/v281_hailo10_nwc/physical_outputs.npz",
    "tests/fixtures/v27931_classification/original_R1_R2_logits.npz",
    "tests/fixtures/v27933_force_origin/force_origin_audit_20260909T095730Z_ni3utrwy.zip",
}
EXCLUDED_FILE_SUFFIXES = {
    ".7z",
    ".bak",
    ".backup",
    ".crdownload",
    ".download",
    ".dxnn",
    ".engine",
    ".gz",
    ".har",
    ".harn",
    ".hef",
    ".log",
    ".npy",
    ".npz",
    ".onnx",
    ".orig",
    ".part",
    ".partial",
    ".parquet",
    ".pyo",
    ".pyc",
    ".rej",
    ".swo",
    ".swp",
    ".tar",
    ".temp",
    ".tgz",
    ".tmp",
    ".whl",
    ".zip",
}

ALLOWED_ROOT_FILES = {
    "IMPLEMENTIERUNGSABGLEICH_2.80.4.md",
    "IMPLEMENTIERUNGSABGLEICH_2.81.md",
    "IMPLEMENTIERUNGSABGLEICH_2.82.md",
    "IMPLEMENTIERUNGSABGLEICH_2.80.1.md",
    "IMPLEMENTIERUNGSABGLEICH_2.80.3.md",
    "PRUEFBERICHT_2.80.1.md",
    "MANIFEST.in",
    "README.md",
    "V2772_HAILO_PARALLEL_CANARY_README.md",
    "analyse_and_split.py",
    "analyse_and_split_gui.py",
    "download_model_zoo_examples.py",
    "model_zoo_manifest.json",
    "pt_to_onnx.py",
    "pyproject.toml",
    "requirements.txt",
    "start_gui.sh",
    "uv.lock",
}
RETAINED_RELEASE_DOC_FILES = {
    "TESTANLEITUNG_2.81.md",
    "VERSION_2.81_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.3.md",
    "TESTANLEITUNG_2.80.4.md",
    "VERSION_2.80.4_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.80.3_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.1.md",
    "TESTANLEITUNG_2.80.2.md",
    "VERSION_2.80.2_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.80.1_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.md",
    "VERSION_2.80_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.31.md",
    "VERSION_2.79.31_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.32.md",
    "VERSION_2.79.32_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.33.md",
    "VERSION_2.79.33_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.34.md",
    "VERSION_2.79.34_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.29.md",
    "VERSION_2.79.30_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.30.md",
    "VERSION_2.79.29_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.28.md",
    "VERSION_2.79.28_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.27.md",
    "VERSION_2.79.27_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.42.md",
    "VERSION_2.75.42_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.43.md",
    "VERSION_2.75.43_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.44.md",
    "VERSION_2.75.44_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.45.md",
    "VERSION_2.75.45_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.46.md",
    "VERSION_2.75.46_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.75.47.md",
    "VERSION_2.75.47_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.19.md",
    "VERSION_2.79.19_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.20.md",
    "VERSION_2.79.20_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.21.md",
    "VERSION_2.79.21_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.22.md",
    "VERSION_2.79.22_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.23.md",
    "VERSION_2.79.23_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.24.md",
    "VERSION_2.79.24_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.25.md",
    "VERSION_2.79.25_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.79.26.md",
    "VERSION_2.79.26_BUILD_AND_TEST_REPORT.md",
}
ALLOWED_SOURCE_TREES = {
    "onnx_splitpoint_tool",
    "scripts",
    "tests",
}
SOURCE_OR_INDEX_ROOTS = {
    *ALLOWED_SOURCE_TREES,
    "docs",
    "profiles",
    "SOURCE_MANIFEST.json",
    "SHA256SUMS.txt",
}

# Exact operational roots retained by ``update_source_release.sh``.  They are
# outside the release inventory only for installed-scope verification and only
# when the root entry is a real, non-symlink directory.
PRESERVED_INSTALL_ROOTS = {
    ".venv",
    ".venv-report",
    ".git",
    ".agents",
    ".codex",
    "logs",
    ".install_logs",
    "EvaluationRuns",
    "RemoteBenchmarkRuns",
    "BenchmarkSets",
    "SplitNetworks",
    "Results",
    "EnergyMeasurements",
    "artifact_store",
    "build_evidence",
}

SOURCE_MANIFEST_TOP_LEVEL_FIELDS = {
    "schema",
    "package_version",
    "workflow_version",
    "file_count",
    "files",
}
SOURCE_MANIFEST_ROW_FIELDS = {"path", "size", "sha256"}
ALLOWED_DOC_FILES = {
    "docs/RELEASE_SCOPE_V282.md",
    "docs/V282_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.82_2026-09-13.md",
    "docs/V2804_QUALITAET_UND_ABNAHME.md",
    "docs/RELEASE_SCOPE_V281.md",
    "docs/V281_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.81_2026-09-13.md",
    "docs/RELEASE_SCOPE_V2804.md",
    "docs/V2804_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.80.4_2026-09-12.md",
    "docs/RELEASE_SCOPE_V2802.md",
    "docs/RELEASE_SCOPE_V2803.md",
    "docs/V2803_DIAGNOSTIC_SCOPE.json",
    "docs/V2803_REFERENCE_AND_DIAGNOSTIC_ACCEPTANCE.md",
    "docs/ARTIFACTS_AND_POSTPROCESSING_CONTRACT.md",
    "docs/V27931_QUALITY_UNCERTAINTY_CONTRACT.md",
    "docs/HAILO10_YOLO26_DIAGNOSE_v27931.md",
    "docs/B500_EVIDENCE_RECONCILIATION_v2792.md",
    "docs/CLEAN_RELEASE.md",
    "docs/ENERGY_WINDOW_METHOD_VALIDATION.md",
    "docs/EVALUATION_PROFILES.md",
    "docs/EVALUATION_WORKFLOW.md",
    "docs/HOLDOUT_ADAPTER_PROTOCOL.md",
    "docs/MODEL_PREPARATION.md",
    "docs/NATIVE_THREE_STAGE.md",
    "docs/NATIVE_BOUNDARY_INTERFACE_VALIDATOR.md",
    "docs/NATIVE_HAILORT_TRT_FIFO_FASTPATH.md",
    "docs/NATIVE_PRODUCER_EVALRUN_USAGE.md",
    "docs/PLATFORM_POWER_CONTROL.md",
    "docs/QUICKSTART.md",
    "docs/README.md",
    "docs/VALIDATION_DATASETS.md",
    "docs/VERSIONING.md",
    "docs/V27933_FORCE_START.md",
    "docs/V27933_NACHTLAUF_UND_GPU.md",
}


def _allowlisted(relative: Path, *, package_version: str) -> bool:
    logical = relative.as_posix()
    if len(relative.parts) == 1:
        return (
            logical in ALLOWED_ROOT_FILES
            or logical in RETAINED_RELEASE_DOC_FILES
            or logical in {
                f"TESTANLEITUNG_{package_version}.md",
                f"VERSION_{package_version}_BUILD_AND_TEST_REPORT.md",
            }
        )
    if relative.parts[0] in ALLOWED_SOURCE_TREES:
        return True
    if relative.parts[0] == "profiles":
        # Only current top-level profiles ship.  Historical legacy_v58 input
        # profiles remain in the development tree but not in clean releases.
        return (
            len(relative.parts) == 2
            and relative.suffix.casefold() in {".yaml", ".yml"}
        )
    if logical in ALLOWED_DOC_FILES:
        return True
    return relative.parts[:2] == ("docs", "calibration")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _source_candidate(
    path: Path,
    root: Path,
    *,
    package_version: str = "",
) -> bool:
    relative = path.relative_to(root)
    if not _allowlisted(relative, package_version=package_version):
        return False
    if path.name in EXCLUDED_FILE_NAMES:
        return False
    casefold_name = path.name.casefold()
    if casefold_name in EXCLUDED_CASEFOLD_FILE_NAMES:
        return False
    if (
        path.name.endswith("~")
        or any(
            casefold_name.startswith(prefix)
            for prefix in EXCLUDED_FILE_PREFIXES
        )
    ):
        return False
    if any(
        part.casefold() in EXCLUDED_DIRECTORY_NAMES
        or part.casefold().endswith(".egg-info")
        for part in relative.parts[:-1]
    ):
        return False
    if path.suffix.lower() in EXCLUDED_FILE_SUFFIXES and relative.as_posix() not in ALLOWED_BINARY_TEST_FIXTURES:
        return False
    return True


def _included(
    path: Path,
    root: Path,
    *,
    package_version: str = "",
) -> bool:
    return _source_candidate(
        path,
        root,
        package_version=package_version,
    ) and path.is_file()


def _source_or_index_symlinks(
    root: Path,
    *,
    installed: bool = False,
) -> list[str]:
    """List every symlink outside a real, explicitly ignored directory.

    A name cannot make a link safe: a top-level ``sitecustomize.py`` or
    sourceless ``sitecustomize.pyc`` link is executable by Python before the
    release verifier gets control.  Ignored directory *symlinks* are therefore
    visible too; only contents below actual cache/build directories are pruned.
    """
    if root.is_symlink():
        return ["."]
    found: list[str] = []
    for current_text, directory_names, file_names in os.walk(
        root, topdown=True, followlinks=False
    ):
        current = Path(current_text)
        kept: list[str] = []
        for name in directory_names:
            path = current / name
            relative = path.relative_to(root)
            if path.is_symlink():
                found.append(relative.as_posix())
                continue
            if (
                installed
                and len(relative.parts) == 1
                and name in PRESERVED_INSTALL_ROOTS
            ):
                continue
            if (
                relative.parts[:1] == ("profiles",)
                and len(relative.parts) == 2
            ):
                # Profile subdirectories are policy violations, not ignored
                # caches, and are handled by the inventory.
                kept.append(name)
                continue
            if (
                name.casefold() in EXCLUDED_DIRECTORY_NAMES
                or name.casefold().endswith(".egg-info")
            ):
                continue
            kept.append(name)
        directory_names[:] = kept
        for name in file_names:
            path = current / name
            relative = path.relative_to(root)
            if path.is_symlink():
                found.append(relative.as_posix())
    return sorted(set(found))


def _release_inventory(
    root: Path,
    *,
    package_version: str,
    installed: bool = False,
) -> tuple[
    list[dict[str, Any]],
    list[str],
    list[str],
    list[dict[str, Any]],
]:
    rows: list[dict[str, Any]] = []
    symlinks: list[str] = []
    invalid_paths: list[str] = []
    unexpected_extras: list[dict[str, Any]] = []

    def record_walk_error(error: OSError) -> None:
        raw = Path(str(error.filename or ""))
        try:
            logical = raw.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            logical = raw.as_posix()
        unexpected_extras.append(
            {"path": logical, "kind": "unreadable_directory"}
        )

    for current_text, directory_names, file_names in os.walk(
        root,
        topdown=True,
        onerror=record_walk_error,
        followlinks=False,
    ):
        current = Path(current_text)
        kept_directories: list[str] = []
        for name in sorted(directory_names):
            path = current / name
            relative = path.relative_to(root)
            logical = relative.as_posix()
            if path.is_symlink():
                symlinks.append(logical)
                continue
            if (
                installed
                and len(relative.parts) == 1
                and name in PRESERVED_INSTALL_ROOTS
            ):
                continue
            if (
                relative.parts[:1] == ("profiles",)
                and len(relative.parts) == 2
            ):
                unexpected_extras.append(
                    {"path": logical, "kind": "directory"}
                )
                continue
            if (
                name.casefold() in EXCLUDED_DIRECTORY_NAMES
                or name.casefold().endswith(".egg-info")
            ):
                continue
            kept_directories.append(name)
        directory_names[:] = kept_directories

        for name in sorted(file_names):
            path = current / name
            relative = path.relative_to(root)
            logical = relative.as_posix()
            if path.is_symlink():
                symlinks.append(logical)
                continue
            try:
                path_stat = path.stat(follow_symlinks=False)
            except OSError:
                unexpected_extras.append(
                    {"path": logical, "kind": "unreadable"}
                )
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                unexpected_extras.append(
                    {"path": logical, "kind": "non_regular"}
                )
                continue
            if not _valid_manifest_path(logical):
                invalid_paths.append(logical)
                continue
            if (
                len(relative.parts) == 1
                and relative.name
                in {"SOURCE_MANIFEST.json", "SHA256SUMS.txt"}
            ):
                continue
            row = {
                "path": logical,
                "size": int(path_stat.st_size),
                "sha256": _sha256(path),
            }
            if _source_candidate(
                path,
                root,
                package_version=package_version,
            ):
                rows.append(row)
            else:
                unexpected_extras.append(
                    {**row, "kind": "regular_file"}
                )
    rows.sort(key=lambda row: str(row["path"]))
    unexpected_extras.sort(key=lambda row: str(row["path"]))
    return (
        rows,
        sorted(set(symlinks)),
        sorted(set(invalid_paths)),
        unexpected_extras,
    )


def _files(root: Path, *, package_version: str) -> list[dict[str, Any]]:
    """Return a release inventory, refusing unsafe names and every symlink."""

    rows, symlinks, invalid_paths, unexpected_extras = _release_inventory(
        root, package_version=package_version
    )
    if symlinks:
        raise RuntimeError(
            "source/index symlinks are not permitted: " + ", ".join(symlinks)
        )
    if invalid_paths:
        raise RuntimeError(
            "invalid source manifest paths: " + ", ".join(invalid_paths)
        )
    if unexpected_extras:
        raise RuntimeError(
            "unexpected source entries are not permitted: "
            + ", ".join(
                str(row.get("path") or "") for row in unexpected_extras
            )
        )
    return rows


def _metadata(root: Path) -> tuple[str, str]:
    pyproject = (root / "pyproject.toml").read_text(encoding="utf-8")
    version_match = re.search(r'^version\s*=\s*"([^"]+)"', pyproject, flags=re.MULTILINE)
    identity = (
        root / "onnx_splitpoint_tool" / "release_identity.py"
    ).read_text(encoding="utf-8")
    identity_version_match = re.search(
        r'^VERSION\s*=\s*"([^"]+)"', identity, flags=re.MULTILINE
    )
    workflow_match = re.search(
        r'^BUILD_ID\s*=\s*"([^"]+)"', identity, flags=re.MULTILINE
    )
    if (
        not version_match
        or not identity_version_match
        or not workflow_match
        or version_match.group(1) != identity_version_match.group(1)
    ):
        raise RuntimeError("package or workflow version could not be resolved")
    return version_match.group(1), workflow_match.group(1)


def build(root: Path) -> dict[str, Any]:
    symlinks = _source_or_index_symlinks(root)
    if symlinks:
        raise RuntimeError(
            "source/index symlinks are not permitted: " + ", ".join(symlinks)
        )
    package_version, workflow_version = _metadata(root)
    files = _files(root, package_version=package_version)
    manifest = {
        "schema": "onnx-splitpoint/source-manifest-v1",
        "package_version": package_version,
        "workflow_version": workflow_version,
        "file_count": len(files),
        "files": files,
    }
    (root / "SOURCE_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    (root / "SHA256SUMS.txt").write_text(
        "".join(f"{row['sha256']}  {row['path']}\n" for row in files),
        encoding="utf-8",
    )
    return manifest


def _strict_manifest_contract(
    value: Any,
) -> tuple[bool, dict[str, Any], list[dict[str, Any]]]:
    """Return the exact manifest shape shared by build and runtime gates."""

    if not isinstance(value, dict):
        return False, {}, []
    manifest = dict(value)
    raw_rows = manifest.get("files")
    if not isinstance(raw_rows, list):
        return False, manifest, []
    rows: list[dict[str, Any]] = []
    for raw in raw_rows:
        if not isinstance(raw, dict) or set(raw) != SOURCE_MANIFEST_ROW_FIELDS:
            return False, manifest, []
        row = dict(raw)
        if not _valid_manifest_path(row.get("path")):
            return False, manifest, []
        if type(row.get("size")) is not int or row["size"] < 0:
            return False, manifest, []
        digest = row.get("sha256")
        if not isinstance(digest, str) or re.fullmatch(
            r"[0-9a-f]{64}", digest
        ) is None:
            return False, manifest, []
        rows.append(row)
    paths = [row["path"] for row in rows]
    shape_ok = bool(
        set(manifest) == SOURCE_MANIFEST_TOP_LEVEL_FIELDS
        and manifest.get("schema") == "onnx-splitpoint/source-manifest-v1"
        and type(manifest.get("file_count")) is int
        and manifest.get("file_count") == len(rows)
        and paths == sorted(paths)
        and len(paths) == len(set(paths))
        and {
            "pyproject.toml",
            "onnx_splitpoint_tool/release_identity.py",
        }.issubset(paths)
    )
    return shape_ok, manifest, rows


def _symlink_failure_result(
    root: Path,
    *,
    scope: str,
    symlinks: list[str],
) -> dict[str, Any]:
    """Return a stable failure without opening a symlinked source or index."""

    manifest: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    manifest_path = root / "SOURCE_MANIFEST.json"
    if manifest_path.is_file() and not manifest_path.is_symlink():
        try:
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            _shape, manifest, rows = _strict_manifest_contract(loaded)
        except Exception:
            pass
    expected_paths = {
        str(row.get("path"))
        for row in rows
        if isinstance(row.get("path"), str)
    }
    missing = sorted(path for path in symlinks if path in expected_paths)
    common_checks = {
        "schema_ok": manifest.get("schema")
        == "onnx-splitpoint/source-manifest-v1",
        "manifest_shape_ok": False,
        "package_version_ok": False,
        "workflow_version_ok": False,
        "file_count_ok": False,
        "files_ok": False,
        "sha256sums_ok": False,
        "unexpected_extras_ok": True,
        "symlinks_ok": False,
    }
    if scope == "release":
        return {
            "ok": False,
            "checks": common_checks,
            "file_count": 0,
            "invalid_paths": [],
            "unexpected_extra_count": 0,
            "unexpected_extras": [],
            "symlinks": list(symlinks),
        }
    installed_checks = {
        **common_checks,
        "manifest_rows_ok": False,
        "unexpected_extras_ok": True,
    }
    return {
        "ok": False,
        "scope": "installed",
        "checks": installed_checks,
        "file_count": 0,
        "release_file_count": len(rows),
        "actual_allowlisted_file_count": 0,
        "user_profile_count": 0,
        "user_profiles": [],
        "unexpected_extra_count": 0,
        "unexpected_extras": [],
        "symlink_count": len(symlinks),
        "symlinks": list(symlinks),
        "missing": missing,
        "changed": [],
    }


def _index_entry_failure_result(
    root: Path,
    *,
    scope: str,
    invalid_indexes: list[str],
) -> dict[str, Any]:
    """Fail before reading a missing or non-regular release index."""

    checks = {
        "source_indexes_regular": False,
        "schema_ok": False,
        "manifest_shape_ok": False,
        "package_version_ok": False,
        "workflow_version_ok": False,
        "file_count_ok": False,
        "files_ok": False,
        "sha256sums_ok": False,
        "unexpected_extras_ok": True,
        "symlinks_ok": True,
    }
    result: dict[str, Any] = {
        "ok": False,
        "checks": checks,
        "file_count": 0,
        "invalid_indexes": list(invalid_indexes),
        "invalid_paths": [],
        "unexpected_extra_count": 0,
        "unexpected_extras": [],
        "symlinks": [],
    }
    if scope == "installed":
        result.update(
            {
                "scope": "installed",
                "release_file_count": 0,
                "actual_allowlisted_file_count": 0,
                "user_profile_count": 0,
                "user_profiles": [],
                "symlink_count": 0,
                "missing": [],
                "changed": [],
            }
        )
        checks["manifest_rows_ok"] = False
    return result


def _invalid_release_indexes(root: Path) -> list[str]:
    invalid: list[str] = []
    for name in ("SOURCE_MANIFEST.json", "SHA256SUMS.txt"):
        path = root / name
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except FileNotFoundError:
            # A missing sums projection can still yield a useful complete
            # file-inventory report; the normal sums check marks it false.
            # A missing manifest cannot be parsed at all.
            if name == "SOURCE_MANIFEST.json":
                invalid.append(name)
            continue
        except OSError:
            invalid.append(name)
            continue
        if not stat.S_ISREG(mode):
            invalid.append(name)
    return invalid


def _verify_release(root: Path) -> dict[str, Any]:
    """Verify an exact release tree.

    Keep this path deliberately unchanged in semantics: release staging and
    packaging must continue to reject every allowlisted file that is not in
    the manifest.
    """
    symlinks = _source_or_index_symlinks(root)
    if symlinks:
        return _symlink_failure_result(root, scope="release", symlinks=symlinks)
    invalid_indexes = _invalid_release_indexes(root)
    if invalid_indexes:
        return _index_entry_failure_result(
            root,
            scope="release",
            invalid_indexes=invalid_indexes,
        )
    loaded = json.loads((root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8"))
    manifest_shape_ok, expected, expected_rows = _strict_manifest_contract(
        loaded
    )
    package_version, workflow_version = _metadata(root)
    (
        actual_files,
        inventory_symlinks,
        invalid_paths,
        unexpected_extras,
    ) = _release_inventory(
        root, package_version=package_version
    )
    symlinks = sorted(set(symlinks + inventory_symlinks))
    expected_sums = "".join(
        f"{row['sha256']}  {row['path']}\n"
        for row in expected_rows
    )
    sums_path = root / "SHA256SUMS.txt"
    actual_sums = sums_path.read_bytes() if sums_path.is_file() else None
    checks = {
        "schema_ok": expected.get("schema") == "onnx-splitpoint/source-manifest-v1",
        "manifest_shape_ok": manifest_shape_ok,
        "manifest_paths_ok": not invalid_paths,
        "package_version_ok": expected.get("package_version") == package_version,
        "workflow_version_ok": expected.get("workflow_version") == workflow_version,
        "file_count_ok": bool(
            manifest_shape_ok and len(expected_rows) == len(actual_files)
        ),
        "files_ok": bool(manifest_shape_ok and expected_rows == actual_files),
        "sha256sums_ok": bool(
            manifest_shape_ok and actual_sums == expected_sums.encode("utf-8")
        ),
        "unexpected_extras_ok": not unexpected_extras,
        "symlinks_ok": not symlinks,
    }
    return {
        "ok": all(checks.values()),
        "checks": checks,
        "file_count": len(actual_files),
        "invalid_paths": invalid_paths,
        "unexpected_extra_count": len(unexpected_extras),
        "unexpected_extras": unexpected_extras,
        "symlinks": symlinks,
    }


def _valid_manifest_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    logical = PurePosixPath(value)
    return (
        not logical.is_absolute()
        and "." not in logical.parts
        and ".." not in logical.parts
        and logical.as_posix() == value
    )


def _installed_inventory(
    root: Path,
    *,
    package_version: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Return every visible file/link outside explicit cache/build trees."""
    files, symlinks, invalid_paths, unexpected_extras = _release_inventory(
        root,
        package_version=package_version,
        installed=True,
    )
    unexpected_extras.extend(
        {"path": path, "kind": "invalid_path"}
        for path in invalid_paths
    )
    unexpected_extras.sort(key=lambda row: str(row["path"]))
    return files, symlinks, unexpected_extras


def _is_unmanifested_user_profile(path: str) -> bool:
    logical = PurePosixPath(path)
    return (
        len(logical.parts) == 2
        and logical.parts[0] == "profiles"
        and logical.suffix.casefold() in {".yaml", ".yml"}
    )


def _unmanifested_profile_policy_extras(
    root: Path,
    *,
    expected_paths: set[str],
) -> tuple[list[dict[str, Any]], list[str]]:
    """Find profile entries outside the one admitted installed-tree shape."""
    profile_root = root / "profiles"
    if profile_root.is_symlink():
        return [], ["profiles"]
    if not profile_root.is_dir():
        return [], []

    extras: list[dict[str, Any]] = []
    symlinks: list[str] = []
    for path in sorted(profile_root.iterdir(), key=lambda item: item.as_posix()):
        relative = path.relative_to(root).as_posix()
        if relative in expected_paths:
            continue
        if path.is_symlink():
            symlinks.append(relative)
            continue
        if path.is_file() and _is_unmanifested_user_profile(relative):
            continue
        if path.is_file():
            extras.append({
                "path": relative,
                "kind": "regular_file",
                "size": path.stat().st_size,
                "sha256": _sha256(path),
            })
        elif path.is_dir():
            extras.append({"path": relative, "kind": "directory"})
        else:
            extras.append({"path": relative, "kind": "non_regular"})
    return extras, symlinks


def _verify_installed(root: Path) -> dict[str, Any]:
    """Verify release files while admitting explicit installed-tree state.

    The updater intentionally retains real top-level operational directories
    and user-created top-level YAML profiles.  Neither becomes part of the
    release inventory.  Every other extra and every non-pruned symlink is
    blocking.
    """
    preflight_symlinks = _source_or_index_symlinks(root, installed=True)
    if preflight_symlinks:
        return _symlink_failure_result(
            root,
            scope="installed",
            symlinks=preflight_symlinks,
        )
    invalid_indexes = _invalid_release_indexes(root)
    if invalid_indexes:
        return _index_entry_failure_result(
            root,
            scope="installed",
            invalid_indexes=invalid_indexes,
        )
    loaded = json.loads(
        (root / "SOURCE_MANIFEST.json").read_text(encoding="utf-8")
    )
    manifest_shape_ok, expected, expected_rows = _strict_manifest_contract(
        loaded
    )
    package_version, workflow_version = _metadata(root)
    manifest_rows_ok = manifest_shape_ok
    expected_paths = [row["path"] for row in expected_rows]
    manifest_paths_unique = len(expected_paths) == len(set(expected_paths))
    expected_by_path = {
        row["path"]: row
        for row in expected_rows
    } if manifest_paths_unique else {}

    actual_files, symlinks, inventory_extras = _installed_inventory(
        root,
        package_version=package_version,
    )
    symlinks = sorted(set(symlinks))

    actual_by_path = {row["path"]: row for row in actual_files}
    actual_owned_rows = [
        row for row in actual_files
        if row["path"] in expected_by_path
    ]
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    changed = []
    for path in sorted(set(expected_by_path) & set(actual_by_path)):
        expected_row = expected_by_path[path]
        actual_row = actual_by_path[path]
        if expected_row != actual_row:
            changed.append({
                "path": path,
                "expected_size": expected_row["size"],
                "actual_size": actual_row["size"],
                "expected_sha256": expected_row["sha256"],
                "actual_sha256": actual_row["sha256"],
            })

    extra_rows = [
        row for row in actual_files
        if row["path"] not in expected_by_path
    ]
    user_profiles = [
        row for row in extra_rows
        if _is_unmanifested_user_profile(row["path"])
    ]
    unexpected_extras = [
        row for row in extra_rows
        if not _is_unmanifested_user_profile(row["path"])
    ]
    unexpected_extras.extend(inventory_extras)
    unexpected_extras.sort(key=lambda row: row["path"])

    canonical_sums = "".join(
        f"{row['sha256']}  {row['path']}\n"
        for row in expected_rows
    )
    sums_path = root / "SHA256SUMS.txt"
    actual_sums = (
        sums_path.read_bytes()
        if sums_path.is_file()
        else None
    )
    release_files_ok = (
        manifest_rows_ok
        and manifest_paths_unique
        and expected_rows == actual_owned_rows
    )
    checks = {
        "schema_ok": expected.get("schema") == "onnx-splitpoint/source-manifest-v1",
        "manifest_shape_ok": manifest_shape_ok,
        "package_version_ok": expected.get("package_version") == package_version,
        "workflow_version_ok": expected.get("workflow_version") == workflow_version,
        "manifest_rows_ok": manifest_rows_ok and manifest_paths_unique,
        "file_count_ok": (
            manifest_rows_ok
            and expected.get("file_count") == len(expected_rows)
            and len(expected_rows) == len(actual_owned_rows)
        ),
        "files_ok": release_files_ok,
        "sha256sums_ok": actual_sums == canonical_sums.encode("utf-8"),
        "unexpected_extras_ok": not unexpected_extras,
        "symlinks_ok": not symlinks,
    }
    return {
        "ok": all(checks.values()),
        "scope": "installed",
        "checks": checks,
        "file_count": len(actual_owned_rows),
        "release_file_count": len(expected_rows),
        "actual_allowlisted_file_count": len(actual_files),
        "user_profile_count": len(user_profiles),
        "user_profiles": user_profiles,
        "unexpected_extra_count": len(unexpected_extras),
        "unexpected_extras": unexpected_extras,
        "symlink_count": len(symlinks),
        "symlinks": symlinks,
        "missing": missing,
        "changed": changed,
    }


def verify(root: Path, *, scope: str = "release") -> dict[str, Any]:
    if scope == "release":
        return _verify_release(root)
    if scope == "installed":
        return _verify_installed(root)
    raise ValueError(f"unsupported source-manifest verification scope: {scope}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=str(Path(__file__).resolve().parents[1]))
    parser.add_argument("--verify", action="store_true")
    parser.add_argument(
        "--scope",
        choices=("release", "installed"),
        default="release",
        help=(
            "verification scope: exact release inventory (default), or "
            "installed tree with separately reported user profiles"
        ),
    )
    args = parser.parse_args()
    root = Path(args.root).expanduser().resolve()
    if not args.verify and args.scope != "release":
        parser.error("--scope installed requires --verify")
    result = verify(root, scope=args.scope) if args.verify else build(root)
    print(json.dumps(result if args.verify else {
        "ok": True,
        "file_count": result["file_count"],
        "package_version": result["package_version"],
        "workflow_version": result["workflow_version"],
    }, indent=2))
    return 0 if result.get("ok", True) else 1


if __name__ == "__main__":
    raise SystemExit(main())
