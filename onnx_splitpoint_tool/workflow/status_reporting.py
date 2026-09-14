"""Display-only status details; never change measurement or acceptance gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

from .evidence_status import project_native_evidence_status


def energy_axis_description(evidence: Mapping[str, Any]) -> str:
    projection = project_native_evidence_status(evidence)
    status = str(projection.get("energy_execution_status") or "unavailable")
    if projection.get("energy_requested") is False:
        return status
    success = projection.get("energy_measurement_success_count")
    planned = projection.get("energy_plan_denominator_count")
    expected = projection.get("energy_matrix_denominator_count")
    if any(value is None for value in (success, planned, expected)):
        return status + "; plan completion / matrix coverage unavailable"
    matrix_complete = bool(expected > 0 and success == expected)
    # The legacy energy status says whether the admitted plan finished. An
    # excluded producer is still part of the requested matrix denominator.
    return (
        f"plan {success}/{planned} successful ({status}); "
        f"matrix {success}/{expected} measured "
        f"({'complete' if matrix_complete else 'incomplete'}); "
        f"excluded {projection.get('energy_plan_excluded_count', 'unavailable')}"
    )


def blocking_reasons_for_display(summary: Mapping[str, Any]) -> list[Any]:
    """An explicit empty blocking list wins over legacy warning aliases."""
    if isinstance(summary.get("blocking_reasons"), list):
        return list(summary["blocking_reasons"])
    return list(summary.get("reasons") or summary.get("partial_reasons") or [])


def deepx_full_primary_failure(run_dir: Path, model_id: str) -> dict[str, str]:
    """Read the original Full execution failure from this run only.

    Repetition aggregation can retain only the later missing-feed error. The
    original result is already an ordinary debug-pack member; no logs, cache
    artifacts or results from another run are searched or reinterpreted.
    """
    if not model_id or Path(model_id).name != model_id:
        return {}
    source = run_dir / "models" / model_id / "benchmark_results" / "benchmark_results_deepx_m1_full_auto.json"
    try:
        if source.resolve().is_relative_to(run_dir.resolve()) is False:
            return {}
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return {}
    rows = payload if isinstance(payload, list) else payload.get("results", []) if isinstance(payload, dict) else []
    failures = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        if str(row.get("backend") or "") != "deepx_m1" or str(row.get("variant") or "") != "full":
            continue
        if row.get("runtime_ok") is not False:
            continue
        prepared = row.get("deepx_prepared_feed_benchmark")
        prepared = prepared if isinstance(prepared, Mapping) else {}
        reason = str(prepared.get("error") or row.get("error_detail") or row.get("error") or "").strip()
        if reason:
            failures.append(reason)
    if not failures or len(set(failures)) != 1:
        return {}
    return {"primary_failure_reason": failures[0], "primary_failure_source": str(source.relative_to(run_dir))}
