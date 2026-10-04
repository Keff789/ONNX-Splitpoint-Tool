"""Display-only status details; never change measurement or acceptance gates."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping
from collections import Counter

from .evidence_status import project_native_evidence_status


def energy_attempt_failure(run_dir: Path, attempt_dir: Path | None = None) -> dict[str, Any]:
    """Display the actual pre-measurement failure without replacing old results.

    Session order is the workflow's existing authority; file modification times
    and unrelated diagnostic replays never select the latest measurement attempt.
    """
    run_dir = Path(run_dir)
    if attempt_dir is None:
        try:
            manifest = json.loads((run_dir / "run_manifest.json").read_text())
        except (OSError, ValueError):
            return {}
        for session in reversed(manifest.get("execution_sessions", [])):
            sid = str(session.get("session_id") or "")
            if not sid or Path(sid).name != sid or sid in {".", ".."}:
                continue
            candidate = run_dir / "reports/native_energy_measurements/attempts" / sid
            if (candidate / "native_producer_energy_results.json").is_file():
                return {}
            if (candidate / "resume_failure.json").is_file():
                attempt_dir = candidate
                break
    if attempt_dir is None:
        return {}
    source = Path(attempt_dir) / "resume_failure.json"
    try:
        if not source.resolve().is_relative_to(run_dir.resolve()):
            return {}
        payload = json.loads(source.read_text())
    except (OSError, ValueError):
        return {}
    preparation = payload.get("resume_preparation") or {}
    cohort = preparation.get("cohort_preflight") or {}
    failures = Counter()
    row_failures = []
    for row in cohort.get("rows", []):
        if row.get("ok") is True:
            continue
        reason = str(row.get("reason") or "unknown")
        attestation = Path(str((row.get("preflight") or {}).get("attestation_path") or ""))
        try:
            if attestation.resolve().is_relative_to(Path(attempt_dir).resolve()):
                evidence = json.loads(attestation.read_text())
                reason = str(evidence.get("failure_reason") or evidence.get("error") or evidence.get("reason") or reason)
        except (OSError, ValueError):
            pass
        if reason in {"preflight_evidence_not_verified", "unknown"}:
            # Full artifact verification can exit with structured stderr
            # before an attestation exists. Preserve that first concrete error.
            stderr = Path(str((row.get("preflight") or {}).get("stderr_path") or ""))
            try:
                if stderr.resolve().is_relative_to(Path(attempt_dir).resolve()):
                    with stderr.open() as stream:
                        text = stream.read(16384)
                    for line in reversed(text.splitlines()):
                        try:
                            evidence = json.loads(line)
                        except ValueError:
                            continue
                        if isinstance(evidence, dict) and evidence.get("ok") is False:
                            reason = str(evidence.get("status") or reason)
                            break
            except OSError:
                pass
        row_failures.append({"identity": row.get("identity"), "reason": reason,
                             "evidence": str(attestation)})
        failures[reason.removeprefix("RuntimeError:").split(":/", 1)[0]] += 1
    counters = {key: payload.get(key, 0) for key in (
        "measurement_wrapper_started_count", "started_measurement_count",
        "collector_started_repeat_count", "workload_started_repeat_count")}
    counts = {key: cohort.get(key, 0) for key in (
        "preflight_attempted_count", "preflight_passed_count", "preflight_failed_count")}
    detail = (f"{payload.get('status', 'failed')}: "
              f"artifact-only preflight {counts['preflight_passed_count']}/{counts['preflight_attempted_count']} PASS, "
              f"{counts['preflight_failed_count']} FAIL; "
              + "; ".join(f"{n} × {reason}" for reason, n in sorted(failures.items()))
              + f"; starts wrapper/collector/workload={counters['measurement_wrapper_started_count']}/"
              f"{counters['collector_started_repeat_count']}/{counters['workload_started_repeat_count']}; {source}")
    return {"status": payload.get("status", "failed"), "source": str(source),
            "attempt_id": Path(attempt_dir).name, **counts, **counters,
            "failure_counts": dict(failures), "row_failures": row_failures,
            "energy_not_started_categories": ["native_runtime_preflight"],
            "energy_not_started_reason": detail}


def energy_axis_description(evidence: Mapping[str, Any]) -> str:
    projection = project_native_evidence_status(evidence)
    status = str(projection.get("energy_execution_status") or "unavailable")
    if projection.get("energy_requested") is False:
        return "not_requested"
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
