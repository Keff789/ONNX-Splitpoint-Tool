from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkDirLayout:
    root: Path
    split_networks: Path
    benchmark_sets: Path
    results: Path


def ensure_workdir(root: Path) -> WorkDirLayout:
    """Ensure the working directory structure exists.

    Layout (under root):
      SplitNetworks/
      BenchmarkSets/
      Results/
    """
    root = Path(root).expanduser().resolve()
    split_networks = root / "SplitNetworks"
    benchmark_sets = root / "BenchmarkSets"
    results = root / "Results"
    split_networks.mkdir(parents=True, exist_ok=True)
    benchmark_sets.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    return WorkDirLayout(root=root, split_networks=split_networks, benchmark_sets=benchmark_sets, results=results)
