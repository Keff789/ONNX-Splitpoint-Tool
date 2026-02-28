"""Local helpers for bundling benchmark results.

The remote benchmark pipeline collects artifacts from a suite directory into a
dedicated results directory, and then tars it. On remote hosts this is done
with shell commands.

This module provides a *Python* implementation of the same logic to:
  - make the behavior testable (contract/guardrail tests)
  - keep artifact expectations documented in code
"""

from __future__ import annotations

import shutil
import tarfile
from pathlib import Path
from typing import List


SUITE_LEVEL_PATTERNS: List[str] = [
    "benchmark_results_*.json",
    "benchmark_results_*.csv",
    "benchmark_summary_*.md",
    "benchmark_table_*.tex",
    "analysis_*.pdf",
    "analysis_*.svg",
]


CASE_LEVEL_PATTERNS_FILES: List[str] = [
    "benchmark_*",
    "paper_figures_*",
]


CASE_LEVEL_PATTERNS_DIRS: List[str] = [
    "results_*",
    "validation_*",
]


def collect_results_from_suite(suite_dir: Path, dst_results_dir: Path) -> None:
    """Collect the relevant artifacts from a suite directory.

    The resulting directory structure is compatible with the remote
    ``results_bundle.tar.gz`` layout.
    """

    suite_dir = Path(suite_dir)
    dst_results_dir = Path(dst_results_dir)
    dst_results_dir.mkdir(parents=True, exist_ok=True)

    # Suite-level artifacts.
    for pat in SUITE_LEVEL_PATTERNS:
        for p in suite_dir.glob(pat):
            if p.is_file():
                shutil.copy2(p, dst_results_dir / p.name)

    # Per-case artifacts.
    for case_dir in sorted([p for p in suite_dir.iterdir() if p.is_dir()]):
        cid = case_dir.name
        out_case = dst_results_dir / cid
        out_case.mkdir(parents=True, exist_ok=True)

        # Shallow files.
        for pat in CASE_LEVEL_PATTERNS_FILES:
            for p in case_dir.glob(pat):
                if p.is_file():
                    shutil.copy2(p, out_case / p.name)

        # Directories (results_*/validation_*).
        for pat in CASE_LEVEL_PATTERNS_DIRS:
            for p in case_dir.glob(pat):
                if p.is_dir():
                    shutil.copytree(p, out_case / p.name, dirs_exist_ok=True)


def create_results_bundle(results_dir: Path, tar_gz_path: Path) -> None:
    """Create a .tar.gz bundle from a collected results directory."""

    results_dir = Path(results_dir)
    tar_gz_path = Path(tar_gz_path)
    tar_gz_path.parent.mkdir(parents=True, exist_ok=True)

    with tarfile.open(tar_gz_path, "w:gz") as tf:
        # Use '.' arcname to match the remote bundling behavior.
        tf.add(results_dir, arcname=".")
