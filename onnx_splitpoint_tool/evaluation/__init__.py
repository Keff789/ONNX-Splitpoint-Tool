from __future__ import annotations

from pathlib import Path
from typing import Callable

from . import calibration, compare, normalize, reporting


def evaluate_results_vs_predictions(
    results_dir: str | Path,
    *,
    suite_dir: str | Path | None = None,
    out_dir: str | Path | None = None,
    log: Callable[[str], None] | None = None,
) -> dict:
    """Run the local prediction-vs-measurement evaluation pipeline."""

    bundle = normalize.load_evaluation_inputs(results_dir, suite_dir=suite_dir, log=log)
    comparison = compare.compare_predictions_vs_measurements(bundle)
    calibration_bundle = calibration.fit_backend_calibration(
        comparison.get("rows") or [],
        suite_meta=comparison.get("suite_meta") or {},
    )
    target_dir = Path(out_dir).expanduser() if out_dir else Path(bundle["run_dir"]) / "evaluation"
    return reporting.write_evaluation_outputs(
        comparison,
        calibration_bundle=calibration_bundle,
        out_dir=target_dir,
        log=log,
    )


__all__ = ["evaluate_results_vs_predictions", "calibration", "compare", "normalize", "reporting"]