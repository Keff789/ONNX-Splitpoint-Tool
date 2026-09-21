"""Strict runtime verification of the installed release source inventory.

The build-time manifest script is intentionally not imported here: runtime
claim admission needs a package-owned verifier that is available in every
installed process.  ``SOURCE_MANIFEST.json`` is the release inventory and
``SHA256SUMS.txt`` is required to be its exact textual projection.  Installed
trees may additionally contain the updater's explicitly preserved operational
directories and user-preserved top-level YAML profiles.  Neither becomes part
of the release inventory.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import stat
from pathlib import Path, PurePosixPath
from typing import Any, Mapping

from .release_identity import BUILD_ID, VERSION


SOURCE_MANIFEST_SCHEMA = "onnx-splitpoint/source-manifest-v1"
SOURCE_INTEGRITY_REPORT_SCHEMA = (
    "onnx-splitpoint/installed-source-integrity-verification"
)
SOURCE_INTEGRITY_BINDING_SCHEMA = (
    "onnx-splitpoint/installed-source-integrity-binding"
)
SOURCE_MANIFEST_NAME = "SOURCE_MANIFEST.json"
SOURCE_SUMS_NAME = "SHA256SUMS.txt"

# Keep this set identical to the updater's trusted installed-tree scanner.
# Only a real directory with one of these exact names at the installation root
# is operational state.  A symlink, regular file or nested homonym remains a
# blocking source-integrity violation.
_PRESERVED_INSTALL_ROOTS = {
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
_REPORT_SAMPLE_LIMIT = 100

_EXCLUDED_DIRECTORY_NAMES = {
    ".cache",
    ".git",
    ".hypothesis",
    ".mypy_cache",
    ".nox",
    ".pytest_cache",
    ".ruff_cache",
    ".tox",
    ".venv",
    "__pycache__",
    "build",
    "dist",
    "htmlcov",
}
_EXCLUDED_FILE_NAMES = {".git", SOURCE_MANIFEST_NAME, SOURCE_SUMS_NAME}
_EXCLUDED_CASEFOLD_FILE_NAMES = {
    ".coverage",
    ".directory",
    ".ds_store",
    "coverage.json",
    "coverage.xml",
    "desktop.ini",
    "thumbs.db",
}
_EXCLUDED_FILE_PREFIXES = (".#", "._", ".coverage.")
_ALLOWED_BINARY_TEST_FIXTURES = {
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/repeat_retry_attempts/repeat_000/attempt_01/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_001/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26m/measurement/run_002/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/repeat_retry_attempts/repeat_000/attempt_01/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_000/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_001/workload_stdout.log',
    'tests/fixtures/v282_energy_retry_original/yolo26s/measurement/run_002/workload_stdout.log',
    'tests/fixtures/v282_h10_generic/models/yolo26m/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log',
    'tests/fixtures/v282_h10_generic/models/yolo26s/benchmark_results/remote_diagnostics/orin_nx_hailo10_01/logs/runner.log',

    "tests/fixtures/v281_hailo10_nwc/physical_outputs.npz",
    "tests/fixtures/v27930/semantic_probe_outputs.npz",
    "tests/fixtures/v27931_classification/original_R1_R2_logits.npz",
    "tests/fixtures/v27933_force_origin/force_origin_audit_20260909T095730Z_ni3utrwy.zip",
}
_EXCLUDED_FILE_SUFFIXES = {
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
    ".pyc",
    ".pyo",
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
_ALLOWED_ROOT_FILES = {
    ".gitignore",
    "AGENTS.md",
    "LICENSE",
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
_RETAINED_RELEASE_DOC_FILES = {
    "TESTANLEITUNG_2.81.md",
    "TESTANLEITUNG_2.82.md",
    "VERSION_2.82_BUILD_AND_TEST_REPORT.md",
    "VERSION_2.81_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.4.md",
    "VERSION_2.80.4_BUILD_AND_TEST_REPORT.md",
    "TESTANLEITUNG_2.80.3.md",
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
_ALLOWED_SOURCE_TREES = {"onnx_splitpoint_tool", "scripts", "tests"}
_ALLOWED_DOC_FILES = {
    "docs/RELEASE_SCOPE_V282.md",
    "docs/V282_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.82_2026-09-13.md",
    "docs/RELEASE_2.83.md",
    "docs/RELEASE_2.90.0.md",
    "docs/RELEASE_2.90.1.md",
    "docs/ARBEITSSTAND.md",
    "docs/RELEASE_SCOPE_V281.md",
    "docs/V281_EVIDENCE_INDEX.json",
    "docs/ONNX_SPLITPOINT_KnowledgeBase_CANONICAL_v2.81_2026-09-13.md",
    "docs/V2804_QUALITAET_UND_ABNAHME.md",
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
    "docs/NATIVE_BOUNDARY_INTERFACE_VALIDATOR.md",
    "docs/NATIVE_HAILORT_TRT_FIFO_FASTPATH.md",
    "docs/NATIVE_PRODUCER_EVALRUN_USAGE.md",
    "docs/NATIVE_THREE_STAGE.md",
    "docs/PLATFORM_POWER_CONTROL.md",
    "docs/QUICKSTART.md",
    "docs/README.md",
    "docs/VALIDATION_DATASETS.md",
    "docs/VERSIONING.md",
    "docs/V27933_FORCE_START.md",
    "docs/V27933_NACHTLAUF_UND_GPU.md",
}
_SOURCE_OR_INDEX_ROOTS = {
    *_ALLOWED_SOURCE_TREES,
    "docs",
    "profiles",
    SOURCE_MANIFEST_NAME,
    SOURCE_SUMS_NAME,
}


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _strict_sha256(value: Any) -> str:
    raw = str(value or "").strip()
    if len(raw) != 64 or raw.lower() != raw:
        return ""
    if any(character not in "0123456789abcdef" for character in raw):
        return ""
    return raw


def _valid_logical_path(value: Any) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    logical = PurePosixPath(value)
    return bool(
        not logical.is_absolute()
        and "." not in logical.parts
        and ".." not in logical.parts
        and logical.as_posix() == value
    )


def _allowlisted(relative: Path, *, package_version: str) -> bool:
    logical = relative.as_posix()
    if len(relative.parts) == 1:
        return bool(
            logical in _ALLOWED_ROOT_FILES
            or logical in _RETAINED_RELEASE_DOC_FILES
            or logical
            in {
                f"TESTANLEITUNG_{package_version}.md",
                f"VERSION_{package_version}_BUILD_AND_TEST_REPORT.md",
            }
        )
    if relative.parts[0] in _ALLOWED_SOURCE_TREES:
        return True
    if relative.parts[0] == "profiles":
        return bool(
            len(relative.parts) == 2
            and relative.suffix.casefold() in {".yaml", ".yml"}
        )
    if logical in _ALLOWED_DOC_FILES:
        return True
    return relative.parts[:2] == ("docs", "calibration")


def _source_candidate(relative: Path, *, package_version: str) -> bool:
    if not _allowlisted(relative, package_version=package_version):
        return False
    if relative.name in _EXCLUDED_FILE_NAMES:
        return False
    casefold_name = relative.name.casefold()
    if casefold_name in _EXCLUDED_CASEFOLD_FILE_NAMES:
        return False
    if relative.name.endswith("~") or any(
        casefold_name.startswith(prefix) for prefix in _EXCLUDED_FILE_PREFIXES
    ):
        return False
    if any(
        part.casefold() in _EXCLUDED_DIRECTORY_NAMES
        or part.casefold().endswith(".egg-info")
        for part in relative.parts[:-1]
    ):
        return False
    return relative.suffix.lower() not in _EXCLUDED_FILE_SUFFIXES or relative.as_posix() in _ALLOWED_BINARY_TEST_FIXTURES


def _is_user_profile(path: str) -> bool:
    logical = PurePosixPath(path)
    return bool(
        len(logical.parts) == 2
        and logical.parts[0] == "profiles"
        and logical.suffix.casefold() in {".yaml", ".yml"}
    )


def _inventory(
    root: Path,
    *,
    package_version: str,
) -> tuple[list[dict[str, Any]], list[str], list[dict[str, Any]]]:
    """Inventory release files outside explicit operational/cache directories.

    File-name and suffix exclusions decide manifest membership; they must not
    make an unmanifested executable input invisible.  In particular,
    top-level ``sitecustomize.py`` and sourceless ``sitecustomize.pyc`` files
    are unexpected extras.  Every symlink is blocking too, including links
    named like an otherwise ignored directory.
    """
    files: list[dict[str, Any]] = []
    symlinks: list[str] = []
    inventory_extras: list[dict[str, Any]] = []

    def record_walk_error(error: OSError) -> None:
        raw = Path(str(error.filename or ""))
        try:
            logical = raw.absolute().relative_to(root.absolute()).as_posix()
        except ValueError:
            logical = raw.as_posix()
        inventory_extras.append(
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
                len(relative.parts) == 1
                and name in _PRESERVED_INSTALL_ROOTS
            ):
                # ``os.walk`` classified this entry as a directory and the
                # symlink check above already rejected linked roots.  Prune the
                # complete operational subtree without opening or hashing it.
                continue
            if (
                relative.parts
                and relative.parts[0] == "profiles"
                and len(relative.parts) == 2
            ):
                inventory_extras.append(
                    {"path": logical, "kind": "directory"}
                )
                continue
            if (
                name.casefold() in _EXCLUDED_DIRECTORY_NAMES
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
                inventory_extras.append(
                    {"path": logical, "kind": "unreadable"}
                )
                continue
            if not stat.S_ISREG(path_stat.st_mode):
                inventory_extras.append(
                    {"path": logical, "kind": "non_regular"}
                )
                continue
            if (
                len(relative.parts) == 1
                and relative.name in _PRESERVED_INSTALL_ROOTS
            ):
                inventory_extras.append(
                    {
                        "path": logical,
                        "size": int(path_stat.st_size),
                        "kind": "preserved_install_root_not_directory",
                    }
                )
                continue
            if (
                len(relative.parts) == 1
                and relative.name in {SOURCE_MANIFEST_NAME, SOURCE_SUMS_NAME}
            ):
                continue
            if not _source_candidate(
                relative, package_version=package_version
            ):
                # A rejected entry cannot become claim-bearing source.  Its
                # path, type and size are sufficient diagnostics; do not read
                # potentially huge unrelated data merely to reject it.
                inventory_extras.append(
                    {
                        "path": logical,
                        "size": int(path_stat.st_size),
                        "kind": "regular_file",
                    }
                )
                continue
            files.append(
                {
                    "path": logical,
                    "size": int(path_stat.st_size),
                    "sha256": _sha256_file(path),
                }
            )
    files.sort(key=lambda row: str(row["path"]))
    inventory_extras.sort(key=lambda row: str(row["path"]))
    return files, sorted(set(symlinks)), inventory_extras


def _base_report(root: Path) -> dict[str, Any]:
    return {
        "schema": SOURCE_INTEGRITY_REPORT_SCHEMA,
        "schema_version": 1,
        "ok": False,
        "status": "invalid",
        "root": str(root),
        "package_version": VERSION,
        "build_id": BUILD_ID,
        "manifest_path": str(root / SOURCE_MANIFEST_NAME),
        "manifest_sha256": "",
        "sha256sums_path": str(root / SOURCE_SUMS_NAME),
        "sha256sums_sha256": "",
        "checks": {},
        "errors": [],
        "missing": [],
        "changed": [],
        "unexpected_extras": [],
        "unexpected_extra_count": 0,
        "unexpected_extras_truncated": False,
        "user_profiles": [],
        "symlinks": [],
        "symlink_count": 0,
        "symlinks_truncated": False,
    }


def verify_installed_source_integrity(
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Verify the installed release and return its authoritative manifest SHA.

    This function is read-only and does not follow source/index symlinks.
    Failures are returned as structured evidence so GUI and preflight callers
    can explain a refusal without attempting a hardware operation.
    """

    source_root = (
        Path(root).expanduser().absolute()
        if root is not None
        else Path(__file__).resolve().parents[1]
    )
    report = _base_report(source_root)
    errors: list[str] = report["errors"]
    if source_root.is_symlink():
        report["symlinks"] = ["."]
        errors.append("source_root_symlink")
        report["checks"] = {"source_root_regular_directory": False}
        report["status"] = "source_root_symlink"
        return report
    manifest_path = source_root / SOURCE_MANIFEST_NAME
    sums_path = source_root / SOURCE_SUMS_NAME

    index_symlinks = [
        path.name for path in (manifest_path, sums_path) if path.is_symlink()
    ]
    if index_symlinks:
        report["symlinks"] = sorted(index_symlinks)
        errors.append("source_index_symlink")
        report["checks"] = {
            "manifest_regular": False,
            "sha256sums_regular": False,
        }
        report["status"] = "source_index_symlink"
        return report

    for path, label in (
        (manifest_path, "source_manifest"),
        (sums_path, "sha256sums"),
    ):
        try:
            mode = path.stat(follow_symlinks=False).st_mode
        except OSError as exc:
            errors.append(f"{label}_unreadable:{type(exc).__name__}")
            continue
        if not stat.S_ISREG(mode):
            errors.append(f"{label}_not_regular")
    if errors:
        report["status"] = "source_indexes_unreadable"
        return report

    try:
        manifest_bytes = manifest_path.read_bytes()
        manifest_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
        report["manifest_sha256"] = manifest_sha256
        loaded = json.loads(manifest_bytes.decode("utf-8"))
    except Exception as exc:
        errors.append(f"source_manifest_invalid:{type(exc).__name__}")
        report["status"] = "source_manifest_invalid"
        return report
    if not isinstance(loaded, Mapping):
        errors.append("source_manifest_not_mapping")
        report["status"] = "source_manifest_invalid"
        return report

    manifest = dict(loaded)
    top_level_exact = set(manifest) == {
        "schema",
        "package_version",
        "workflow_version",
        "file_count",
        "files",
    }
    raw_rows = manifest.get("files")
    rows_list_ok = isinstance(raw_rows, list)
    rows: list[dict[str, Any]] = []
    row_errors: list[str] = []
    if rows_list_ok:
        for index, value in enumerate(raw_rows):
            if not isinstance(value, Mapping) or set(value) != {
                "path",
                "size",
                "sha256",
            }:
                row_errors.append(f"row_{index}_shape_invalid")
                continue
            row = dict(value)
            if not _valid_logical_path(row.get("path")):
                row_errors.append(f"row_{index}_path_invalid")
            if (
                isinstance(row.get("size"), bool)
                or not isinstance(row.get("size"), int)
                or int(row.get("size")) < 0
            ):
                row_errors.append(f"row_{index}_size_invalid")
            if not _strict_sha256(row.get("sha256")):
                row_errors.append(f"row_{index}_sha256_invalid")
            rows.append(row)
    paths = [str(row.get("path") or "") for row in rows]
    rows_sorted = paths == sorted(paths)
    rows_unique = len(paths) == len(set(paths))
    required_identity_rows = {
        "pyproject.toml",
        "onnx_splitpoint_tool/release_identity.py",
    }.issubset(paths)
    manifest_shape_ok = bool(
        top_level_exact
        and rows_list_ok
        and not row_errors
        and rows_sorted
        and rows_unique
        and required_identity_rows
        and not isinstance(manifest.get("file_count"), bool)
        and isinstance(manifest.get("file_count"), int)
        and int(manifest.get("file_count")) == len(rows)
    )

    actual_files, symlinks, inventory_extras = _inventory(
        source_root,
        package_version=VERSION,
    )
    expected_by_path = {
        str(row["path"]): row for row in rows
    } if manifest_shape_ok else {}
    actual_by_path = {str(row["path"]): row for row in actual_files}
    missing = sorted(set(expected_by_path) - set(actual_by_path))
    changed: list[dict[str, Any]] = []
    for path in sorted(set(expected_by_path) & set(actual_by_path)):
        expected = expected_by_path[path]
        actual = actual_by_path[path]
        if expected != actual:
            changed.append(
                {
                    "path": path,
                    "expected_size": expected.get("size"),
                    "actual_size": actual.get("size"),
                    "expected_sha256": expected.get("sha256"),
                    "actual_sha256": actual.get("sha256"),
                }
            )
    extras = [
        row for row in actual_files if str(row["path"]) not in expected_by_path
    ]
    user_profiles = [
        row for row in extras if _is_user_profile(str(row["path"]))
    ]
    unexpected_extras = [
        row for row in extras if not _is_user_profile(str(row["path"]))
    ]
    unexpected_extras.extend(inventory_extras)
    unexpected_extras.sort(key=lambda row: str(row["path"]))

    canonical_sums = "".join(
        f"{row['sha256']}  {row['path']}\n" for row in rows
    )
    try:
        sums_bytes = sums_path.read_bytes()
        report["sha256sums_sha256"] = hashlib.sha256(sums_bytes).hexdigest()
        sums_ok = sums_bytes == canonical_sums.encode("utf-8")
    except OSError as exc:
        errors.append(f"sha256sums_unreadable:{type(exc).__name__}")
        sums_ok = False

    checks = {
        "manifest_regular": True,
        "sha256sums_regular": True,
        "schema_ok": manifest.get("schema") == SOURCE_MANIFEST_SCHEMA,
        "package_version_ok": manifest.get("package_version") == VERSION,
        "build_id_ok": manifest.get("workflow_version") == BUILD_ID,
        "manifest_shape_ok": manifest_shape_ok,
        "rows_ok": not missing and not changed,
        "sha256sums_ok": sums_ok,
        "unexpected_extras_ok": not unexpected_extras,
        "source_index_symlinks_ok": not symlinks,
    }
    if row_errors:
        errors.extend(row_errors)
    if not rows_sorted:
        errors.append("source_manifest_rows_not_sorted")
    if not rows_unique:
        errors.append("source_manifest_paths_not_unique")
    if not required_identity_rows:
        errors.append("source_manifest_identity_rows_missing")
    if missing:
        errors.append("release_files_missing")
    if changed:
        errors.append("release_files_changed")
    if unexpected_extras:
        errors.append("unexpected_allowlisted_files")
    if symlinks:
        errors.append("source_or_index_symlink")
    if not sums_ok:
        errors.append("sha256sums_mismatch")

    unexpected_extra_count = len(unexpected_extras)
    symlink_count = len(symlinks)
    report.update(
        {
            "checks": checks,
            "missing": missing,
            "changed": changed,
            "unexpected_extras": unexpected_extras[:_REPORT_SAMPLE_LIMIT],
            "unexpected_extra_count": unexpected_extra_count,
            "unexpected_extras_truncated": (
                unexpected_extra_count > _REPORT_SAMPLE_LIMIT
            ),
            "user_profiles": user_profiles,
            "symlinks": symlinks[:_REPORT_SAMPLE_LIMIT],
            "symlink_count": symlink_count,
            "symlinks_truncated": symlink_count > _REPORT_SAMPLE_LIMIT,
            "manifest_file_count": len(rows),
            "verified_release_file_count": len(expected_by_path) - len(missing),
            "user_profile_count": len(user_profiles),
        }
    )
    report["ok"] = all(checks.values()) and not errors
    report["status"] = "verified" if report["ok"] else "source_integrity_failed"
    return report


def create_source_integrity_binding(
    verification_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Seal the authoritative release-index identity and its preflight report."""

    report = copy.deepcopy(dict(verification_report or {}))
    manifest_sha256 = _strict_sha256(report.get("manifest_sha256"))
    sums_sha256 = _strict_sha256(report.get("sha256sums_sha256"))
    if (
        report.get("ok") is not True
        or report.get("status") != "verified"
        or not manifest_sha256
        or not sums_sha256
        or report.get("package_version") != VERSION
        or report.get("build_id") != BUILD_ID
    ):
        raise ValueError("installed source-integrity report is not claim-admissible")
    return {
        "schema": SOURCE_INTEGRITY_BINDING_SCHEMA,
        "schema_version": 1,
        "package_version": VERSION,
        "build_id": BUILD_ID,
        "manifest_path": str(report.get("manifest_path") or ""),
        "manifest_sha256": manifest_sha256,
        "sha256sums_path": str(report.get("sha256sums_path") or ""),
        "sha256sums_sha256": sums_sha256,
        "verification_report": report,
    }


def verify_source_integrity_binding(
    binding: Mapping[str, Any] | None,
    *,
    root: str | Path | None = None,
    current_report: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Reverify this package and exact-match it to a sealed source binding."""

    bound = dict(binding or {})
    current = copy.deepcopy(
        dict(current_report)
        if current_report is not None
        else verify_installed_source_integrity(root)
    )
    errors: list[str] = []
    required_keys = {
        "schema",
        "schema_version",
        "package_version",
        "build_id",
        "manifest_path",
        "manifest_sha256",
        "sha256sums_path",
        "sha256sums_sha256",
        "verification_report",
    }
    if set(bound) != required_keys:
        errors.append("source_integrity_binding_shape_invalid")
    if bound.get("schema") != SOURCE_INTEGRITY_BINDING_SCHEMA:
        errors.append("source_integrity_binding_schema_invalid")
    if type(bound.get("schema_version")) is not int or bound.get(
        "schema_version"
    ) != 1:
        errors.append("source_integrity_binding_version_invalid")
    if bound.get("package_version") != VERSION:
        errors.append("source_integrity_binding_package_version_mismatch")
    if bound.get("build_id") != BUILD_ID:
        errors.append("source_integrity_binding_build_id_mismatch")

    bound_report = (
        dict(bound.get("verification_report") or {})
        if isinstance(bound.get("verification_report"), Mapping)
        else {}
    )
    if (
        bound_report.get("ok") is not True
        or bound_report.get("status") != "verified"
    ):
        errors.append("source_integrity_bound_report_not_verified")
    for field in (
        "package_version",
        "build_id",
        "manifest_path",
        "manifest_sha256",
        "sha256sums_path",
        "sha256sums_sha256",
    ):
        if bound_report.get(field) != bound.get(field):
            errors.append(f"source_integrity_bound_report_{field}_mismatch")

    if current.get("ok") is not True or current.get("status") != "verified":
        errors.append("current_installed_source_integrity_failed")
    for field in (
        "package_version",
        "build_id",
        "manifest_path",
        "manifest_sha256",
        "sha256sums_path",
        "sha256sums_sha256",
    ):
        if current.get(field) != bound.get(field):
            errors.append(f"current_source_{field}_mismatch")

    return {
        "schema": "onnx-splitpoint/source-integrity-binding-verification",
        "schema_version": 1,
        "ok": not errors,
        "status": "verified" if not errors else "source_integrity_binding_failed",
        "errors": list(dict.fromkeys(errors)),
        "binding": copy.deepcopy(bound),
        "current_verification_report": current,
        "manifest_path": str(current.get("manifest_path") or ""),
        "manifest_sha256": str(current.get("manifest_sha256") or ""),
    }


__all__ = [
    "SOURCE_INTEGRITY_BINDING_SCHEMA",
    "SOURCE_INTEGRITY_REPORT_SCHEMA",
    "SOURCE_MANIFEST_NAME",
    "SOURCE_MANIFEST_SCHEMA",
    "SOURCE_SUMS_NAME",
    "create_source_integrity_binding",
    "verify_installed_source_integrity",
    "verify_source_integrity_binding",
]
