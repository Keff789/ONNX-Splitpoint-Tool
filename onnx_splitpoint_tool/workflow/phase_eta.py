from __future__ import annotations

"""Phase-local, cohort-aware and deliberately advisory ETA estimation.

The estimator is designed for heterogeneous worker pools.  It never includes
jobs completed before a phase-local baseline, never pools unrelated cohorts,
and does not print a point estimate before a warm-up population exists.
Direct job durations are interpreted as worker-seconds and divided by the
configured parallelism.  When a result does not expose a duration, observed
wall-clock completion intervals are retained as a separate fallback and are
not divided a second time.
"""

import math
import statistics
import time
from collections import defaultdict, deque
from typing import Any, Mapping


class PhaseEtaEstimator:
    def __init__(
        self,
        *,
        phase: str,
        warmup_completions: int = 3,
        max_samples: int = 128,
        parallelism: int = 1,
    ) -> None:
        self.phase = str(phase)
        self.warmup_completions = max(1, int(warmup_completions))
        self.max_samples = max(8, int(max_samples))
        self.parallelism = max(1, int(parallelism))
        self.started = time.monotonic()
        self.baseline_completed = 0
        self._phase_completion_count = 0
        self._last_completion_at: dict[str, float] = {}
        self._direct_durations: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_samples)
        )
        self._wall_intervals: dict[str, deque[float]] = defaultdict(
            lambda: deque(maxlen=self.max_samples)
        )
        self._counts: dict[str, int] = defaultdict(int)

    def set_baseline(self, completed: int, *, now: float | None = None) -> None:
        """Start a fresh ETA window after precompleted futures were drained."""

        self.baseline_completed = max(0, int(completed))
        self.started = time.monotonic() if now is None else float(now)
        self._phase_completion_count = 0
        self._last_completion_at.clear()
        self._direct_durations.clear()
        self._wall_intervals.clear()
        self._counts.clear()

    def observe_completion(
        self,
        cohort: str,
        *,
        now: float | None = None,
        duration_s: float | None = None,
    ) -> None:
        cohort = str(cohort or "unknown")
        now = time.monotonic() if now is None else float(now)
        previous = self._last_completion_at.get(cohort)
        self._last_completion_at[cohort] = now
        self._counts[cohort] += 1
        self._phase_completion_count += 1

        direct: float | None = None
        try:
            if duration_s is not None:
                direct = float(duration_s)
        except (TypeError, ValueError, OverflowError):
            direct = None
        if direct is not None and math.isfinite(direct) and direct > 0.0:
            self._direct_durations[cohort].append(direct)
        elif previous is not None:
            interval = max(0.0, now - previous)
            if math.isfinite(interval) and interval > 0.0:
                self._wall_intervals[cohort].append(interval)

    @staticmethod
    def _quantiles(samples: list[float]) -> tuple[float, float, float]:
        ordered = sorted(samples)
        median = float(statistics.median(ordered))
        if len(ordered) == 1:
            return median, median, median
        low_idx = max(0, int(math.floor((len(ordered) - 1) * 0.25)))
        high_idx = min(
            len(ordered) - 1,
            int(math.ceil((len(ordered) - 1) * 0.75)),
        )
        return float(ordered[low_idx]), median, float(ordered[high_idx])

    def estimate(
        self, *, remaining_by_cohort: Mapping[str, int]
    ) -> dict[str, Any]:
        lower = 0.0
        upper = 0.0
        details: dict[str, Any] = {}
        unavailable: list[str] = []

        for cohort, remaining_raw in sorted(remaining_by_cohort.items()):
            cohort = str(cohort or "unknown")
            remaining = max(0, int(remaining_raw))
            if remaining == 0:
                details[cohort] = {"remaining": 0, "status": "complete"}
                continue

            direct = list(self._direct_durations.get(cohort, ()))
            wall = list(self._wall_intervals.get(cohort, ()))
            count = int(self._counts.get(cohort, 0))
            if count < self.warmup_completions:
                unavailable.append(cohort)
                details[cohort] = {
                    "remaining": remaining,
                    "status": "warmup",
                    "completion_count": count,
                    "required_completions": self.warmup_completions,
                }
                continue

            if len(direct) >= 2:
                q1, median, q3 = self._quantiles(direct)
                # Direct samples are per-job worker time.  Divide the remaining
                # worker-seconds by the actual pool width exactly once.
                low = min(q1, median) * remaining / self.parallelism
                high = max(q3, median) * remaining / self.parallelism
                source = "direct_job_duration_divided_by_parallelism"
                sample_count = len(direct)
            elif len(wall) >= 2:
                q1, median, q3 = self._quantiles(wall)
                # Completion intervals already express pool wall throughput.
                low = min(q1, median) * remaining
                high = max(q3, median) * remaining
                source = "observed_pool_completion_interval"
                sample_count = len(wall)
            else:
                unavailable.append(cohort)
                details[cohort] = {
                    "remaining": remaining,
                    "status": "insufficient_comparable_samples",
                    "completion_count": count,
                    "direct_sample_count": len(direct),
                    "wall_interval_count": len(wall),
                }
                continue

            # A cohort range is intentionally conservative and never narrower
            # than +/-10 percent around the median projection.
            median_projection = median * remaining
            if source.startswith("direct"):
                median_projection /= self.parallelism
            low = max(0.0, min(low, median_projection * 0.90))
            high = max(high, median_projection * 1.10)
            lower += low
            upper += high
            details[cohort] = {
                "remaining": remaining,
                "status": "available",
                "source": source,
                "sample_count": sample_count,
                "median_sample_s": median,
                "lower_s": low,
                "upper_s": high,
                "parallelism": self.parallelism,
            }

        if unavailable:
            return {
                "status": "unavailable",
                "display": "ETA=UNAVAILABLE",
                "reason": "insufficient_phase_local_comparable_samples",
                "unavailable_cohorts": unavailable,
                "baseline_completed": self.baseline_completed,
                "phase_completion_count": self._phase_completion_count,
                "parallelism": self.parallelism,
                "cohorts": details,
            }
        return {
            "status": "available",
            "lower_s": lower,
            "upper_s": upper,
            "display": f"ETA={lower:.0f}-{upper:.0f}s",
            "baseline_completed": self.baseline_completed,
            "phase_completion_count": self._phase_completion_count,
            "parallelism": self.parallelism,
            "cohorts": details,
        }
