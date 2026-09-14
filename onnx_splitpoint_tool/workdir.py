from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class WorkDirLayout:
    root: Path
    split_networks: Path
    benchmark_sets: Path
    results: Path
    evaluation_runs: Path
    energy_measurements: Path


def ensure_workdir(root: Path) -> WorkDirLayout:
    """Ensure the working directory structure exists.

    Layout (under root):
      SplitNetworks/
      BenchmarkSets/
      Results/
      EvaluationRuns/
      EnergyMeasurements/
    """
    root = Path(root).expanduser().resolve()
    split_networks = root / "SplitNetworks"
    benchmark_sets = root / "BenchmarkSets"
    results = root / "Results"
    evaluation_runs = root / "EvaluationRuns"
    energy_measurements = root / "EnergyMeasurements"
    split_networks.mkdir(parents=True, exist_ok=True)
    benchmark_sets.mkdir(parents=True, exist_ok=True)
    results.mkdir(parents=True, exist_ok=True)
    evaluation_runs.mkdir(parents=True, exist_ok=True)
    energy_measurements.mkdir(parents=True, exist_ok=True)
    return WorkDirLayout(root=root, split_networks=split_networks, benchmark_sets=benchmark_sets, results=results, evaluation_runs=evaluation_runs, energy_measurements=energy_measurements)
