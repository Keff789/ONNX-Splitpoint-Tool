"""Local helpers for bundling benchmark results.

The remote benchmark pipeline collects artifacts from a suite directory into a
dedicated results directory, and then tars it. On remote hosts this is done
with shell commands.

This module provides a *Python* implementation of the same logic to:
  - make the behavior testable (contract/guardrail tests)
  - keep artifact expectations documented in code
  - support both **full** and **lean** results bundles
"""

from __future__ import annotations

import json
import shutil
import tarfile
from pathlib import Path
from typing import Dict, Iterable, List, Literal

from .schema import RESULTS_BUNDLE_SCHEMA, RESULTS_BUNDLE_SCHEMA_VERSION, now_iso


BundleMode = Literal["full", "lean"]


SUITE_LEVEL_PATTERNS: List[str] = [
    "benchmark_results_*.json",
    "benchmark_results_*.csv",
    "benchmark_summary_*.md",
    "benchmark_table_*.tex",
    "benchmark_report_*.md",
    "benchmark_suite_status_matrix.*",
    "benchmark_plan.json",
    "benchmark_set.json",
    "run_meta.json",
    "preflight.json",
    "analysis_*.pdf",
    "analysis_*.svg",
]


<<<<<<< HEAD
LEAN_SUITE_LEVEL_PATTERNS: List[str] = [
    "benchmark_results_*.json",
    "benchmark_results_*.csv",
    "benchmark_summary_*.md",
    "benchmark_table_*.tex",
    "benchmark_plan.json",
    "benchmark_set.json",
    "run_meta.json",
=======
SUITE_LEVEL_DIR_PATTERNS: List[str] = [
    "analysis_tables",
    "analysis_plots",
    "benchmark_tables_*",
    "paper_figures_*",
    "logs",
>>>>>>> 78644e639b71797c8b822fd9296f3cb8d90bfd20
]


CASE_LEVEL_PATTERNS_FILES: List[str] = [
    "benchmark_*",
    "paper_figures_*",
]


LEAN_CASE_LEVEL_PATTERNS_FILES: List[str] = [
    "benchmark_*",
]


CASE_LEVEL_PATTERNS_DIRS: List[str] = [
    "results_*",
    "validation_*",
]


LEAN_CASE_LEVEL_PATTERNS_DIRS: List[str] = [
    "results_*",
]


LEAN_EXCLUDE_GLOBS: List[str] = [
    "**/*.pdf",
    "**/*.svg",
    "**/*.png",
    "**/paper_figures_*",
    "**/detections_*.json",
    "**/detections_*.png",
    "**/classification_*.json",
    "**/classification_*.png",
]


def _matches_any(path: Path, patterns: Iterable[str]) -> bool:
    return any(path.match(pat) for pat in patterns)


def _copy_if_file(src: Path, dst: Path) -> bool:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        return True
    return False


def collect_results_from_suite(suite_dir: Path, dst_results_dir: Path, *, mode: BundleMode = "full") -> Dict[str, int]:
    """Collect the relevant artifacts from a suite directory.

    The resulting directory structure is compatible with the remote
    ``results_bundle.tar.gz`` layout. ``mode='lean'`` keeps compact summary
    artefacts that are sufficient for analysis/review while dropping heavy
    visualization payloads.
    """

    suite_dir = Path(suite_dir)
    dst_results_dir = Path(dst_results_dir)
    dst_results_dir.mkdir(parents=True, exist_ok=True)

    suite_patterns = SUITE_LEVEL_PATTERNS if mode == "full" else LEAN_SUITE_LEVEL_PATTERNS
    case_file_patterns = CASE_LEVEL_PATTERNS_FILES if mode == "full" else LEAN_CASE_LEVEL_PATTERNS_FILES
    case_dir_patterns = CASE_LEVEL_PATTERNS_DIRS if mode == "full" else LEAN_CASE_LEVEL_PATTERNS_DIRS

    copied_files = 0
    copied_dirs = 0

    for pat in suite_patterns:
        for p in suite_dir.glob(pat):
<<<<<<< HEAD
            if _copy_if_file(p, dst_results_dir / p.name):
                copied_files += 1
=======
            if p.is_file():
                shutil.copy2(p, dst_results_dir / p.name)
    for pat in SUITE_LEVEL_DIR_PATTERNS:
        for p in suite_dir.glob(pat):
            if p.is_dir():
                shutil.copytree(p, dst_results_dir / p.name, dirs_exist_ok=True)
>>>>>>> 78644e639b71797c8b822fd9296f3cb8d90bfd20

    for case_dir in sorted([p for p in suite_dir.iterdir() if p.is_dir() and p.name.startswith('b')]):
        cid = case_dir.name
        out_case = dst_results_dir / cid
        out_case.mkdir(parents=True, exist_ok=True)

        for pat in case_file_patterns:
            for p in case_dir.glob(pat):
                if mode == "lean" and _matches_any(Path(cid) / p.name, LEAN_EXCLUDE_GLOBS):
                    continue
                if _copy_if_file(p, out_case / p.name):
                    copied_files += 1

        for pat in case_dir_patterns:
            for p in case_dir.glob(pat):
                if not p.is_dir():
                    continue
                rel = Path(cid) / p.name
                if mode == "lean" and _matches_any(rel, LEAN_EXCLUDE_GLOBS):
                    continue
                shutil.copytree(p, out_case / p.name, dirs_exist_ok=True)
                copied_dirs += 1

    manifest = {
        "schema": RESULTS_BUNDLE_SCHEMA,
        "schema_version": RESULTS_BUNDLE_SCHEMA_VERSION,
        "mode": mode,
        "created_at": now_iso(),
        "suite_dir": str(suite_dir),
        "stats": {"files": copied_files, "dirs": copied_dirs},
    }
    (dst_results_dir / "results_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest["stats"]


def _copy_results_tree_filtered(src_results_dir: Path, dst_results_dir: Path, *, mode: BundleMode = "full") -> Dict[str, int]:
    src_results_dir = Path(src_results_dir)
    dst_results_dir = Path(dst_results_dir)
    dst_results_dir.mkdir(parents=True, exist_ok=True)
    copied_files = 0
    copied_dirs = 0
    for p in src_results_dir.rglob('*'):
        rel = p.relative_to(src_results_dir)
        if p.is_dir():
            continue
        if mode == 'lean' and _matches_any(rel, LEAN_EXCLUDE_GLOBS):
            continue
        dst = dst_results_dir / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(p, dst)
        copied_files += 1
    manifest = {
        "schema": RESULTS_BUNDLE_SCHEMA,
        "schema_version": RESULTS_BUNDLE_SCHEMA_VERSION,
        "mode": mode,
        "created_at": now_iso(),
        "results_dir": str(src_results_dir),
        "stats": {"files": copied_files, "dirs": copied_dirs},
    }
    (dst_results_dir / "results_bundle_manifest.json").write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest["stats"]


def create_results_bundle(results_dir: Path, tar_gz_path: Path, *, mode: BundleMode = "full") -> None:
    """Create a .tar.gz bundle from a collected results directory."""

    results_dir = Path(results_dir)
    tar_gz_path = Path(tar_gz_path)
    tar_gz_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_gz_path, "w:gz") as tf:
<<<<<<< HEAD
        tf.add(results_dir, arcname=".")


def create_results_bundle_from_suite(suite_dir: Path, tar_gz_path: Path, *, mode: BundleMode = "full") -> None:
    """Collect suite results and create a tar.gz bundle in one step."""

    suite_dir = Path(suite_dir)
    tar_gz_path = Path(tar_gz_path)
    tmp_results = tar_gz_path.parent / (tar_gz_path.stem + f"_{mode}_tmp")
    if tmp_results.exists():
        shutil.rmtree(tmp_results, ignore_errors=True)
    tmp_results.mkdir(parents=True, exist_ok=True)
    try:
        collect_results_from_suite(suite_dir, tmp_results, mode=mode)
        create_results_bundle(tmp_results, tar_gz_path, mode=mode)
    finally:
        shutil.rmtree(tmp_results, ignore_errors=True)


def create_results_bundle_from_results_dir(results_dir: Path, tar_gz_path: Path, *, mode: BundleMode = "full") -> None:
    """Create a bundle from an already-collected local results directory."""

    results_dir = Path(results_dir)
    tar_gz_path = Path(tar_gz_path)
    tmp_results = tar_gz_path.parent / (tar_gz_path.stem + f"_{mode}_tmp")
    if tmp_results.exists():
        shutil.rmtree(tmp_results, ignore_errors=True)
    tmp_results.mkdir(parents=True, exist_ok=True)
    try:
        _copy_results_tree_filtered(results_dir, tmp_results, mode=mode)
        create_results_bundle(tmp_results, tar_gz_path, mode=mode)
    finally:
        shutil.rmtree(tmp_results, ignore_errors=True)
=======
        # Use '.' arcname to match the remote bundling behavior.
        tf.add(results_dir, arcname=".")
>>>>>>> 78644e639b71797c8b822fd9296f3cb8d90bfd20
