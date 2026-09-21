"""Request lifecycle presentation, independent of metric/cache identities.

Cancellation is evidence attached at the controlled service shutdown boundary,
never inferred from a later workflow cancellation or an exception name alone.
"""
from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
import time
from typing import Any, Mapping, Sequence


def cancel_context(run_id: str, reason: str = "user_requested") -> dict[str, Any]:
    return {
        "run_id": str(run_id), "reason": str(reason),
        "requested_at": datetime.now(timezone.utc).isoformat(),
        "requested_monotonic": time.monotonic(),
    }


def stamp_exception(exc: BaseException, context: Mapping[str, Any] | None = None) -> BaseException:
    # The first observation is authoritative: later cancellation cannot retime
    # an already observed failure into the shutdown window.
    if not hasattr(exc, "quality_observed_monotonic"):
        exc.quality_observed_monotonic = time.monotonic()
        exc.quality_observed_at = datetime.now(timezone.utc).isoformat()
        exc.quality_cancel_context = dict(context or {})
    return exc


def exception_outcome(exc: BaseException, *, run_id: str) -> dict[str, Any]:
    stamp_exception(exc)
    context = dict(getattr(exc, "quality_cancel_context", {}) or {})
    observed = float(getattr(exc, "quality_observed_monotonic", 0.0))
    requested = float(context.get("requested_monotonic") or 0.0)
    shutdown = float(context.get("shutdown_monotonic") or 0.0)
    name = type(exc).__name__
    cancelled = bool(
        str(run_id) and context.get("run_id") == str(run_id)
        and context.get("reason") and 0 < requested <= shutdown <= observed
        and name in {"CancelledError", "QualityServiceClosedError"}
    )
    return {
        "status": "cancelled" if cancelled else "failed",
        "technical_status": "cancelled" if cancelled else "failed",
        "decision": "unavailable", "quality_decision": "not_evaluated",
        "scientific_status": "unavailable",
        "completion_reason": (
            "service_closed_after_cancel" if cancelled and name == "QualityServiceClosedError"
            else "user_cancelled" if cancelled else "technical_error"
        ),
        "error": f"{name}: {exc}", "exception_type": name,
        "failure_observed_at": getattr(exc, "quality_observed_at", ""),
        "failure_observed_monotonic": observed,
        "cancel_context": context if cancelled else {},
    }


def completed_outcome(result: Mapping[str, Any]) -> dict[str, Any]:
    out = dict(result)
    if out.get("status", "completed") == "completed":
        out.setdefault("completion_reason", "reused_result" if out.get("cache_hit") else "evaluated")
        out.setdefault("quality_decision", out.get("decision", "not_evaluated"))
    return out


def summarize_requests(results: Sequence[Mapping[str, Any]], *, queued: int = 0, running: int = 0) -> dict[str, Any]:
    """Disjoint request counts. Companions remain separate caller identities."""
    counts: Counter[str] = Counter()
    decisions: Counter[str] = Counter()
    uncertainty: Counter[str] = Counter()
    for result in results:
        status = str(result.get("technical_status") or result.get("status") or "failed").lower()
        decision = str(result.get("decision") or result.get("quality_decision") or result.get("task_quality_decision") or "").lower()
        if status in {"completed", "ok", "success"} and decision in {"pass", "fail", "inconclusive", "reference_close", "accuracy_loss", "not_estimable"}:
            counts["evaluated"] += 1
            decisions[decision] += 1
            assessment = result.get("accuracy_assessment") or {}
            if assessment.get("uncertainty"):
                uncertainty[str(assessment["uncertainty"])] += 1
        elif status == "cancelled":
            counts["cancelled"] += 1
        else:
            counts["technical_failed"] += 1
    return {
        "request_count": len(results) + int(queued) + int(running),
        "terminal_count": len(results), "evaluated_count": counts["evaluated"],
        "completed_count": counts["evaluated"], "cancelled_count": counts["cancelled"],
        "technical_failed_count": counts["technical_failed"],
        "queued_count": int(queued), "running_count": int(running),
        "quality_decision_counts": {name: decisions[name] for name in ("pass", "fail", "inconclusive", "reference_close", "accuracy_loss", "not_estimable")},
        "quality_uncertainty_counts": dict(uncertainty),
    }


def progress_text(counts: Mapping[str, Any]) -> str:
    decisions = counts.get("quality_decision_counts") or {}
    if any(decisions.get(k, 0) for k in ("reference_close", "accuracy_loss", "not_estimable")):
        return (f"{counts['evaluated_count']}/{counts['request_count']} ausgewertet; "
                f"{counts['technical_failed_count']} technische Fehler; "
                f"{decisions.get('reference_close', 0)} Referenznah / {decisions.get('accuracy_loss', 0)} Genauigkeitsverlust / "
                f"{decisions.get('not_estimable', 0)} nicht einstufbar; "
                f"davon {(counts.get('quality_uncertainty_counts') or {}).get('inconclusive', 0)} statistisch unsicher")
    return (
        f"{counts['evaluated_count']}/{counts['request_count']} evaluated; "
        f"{counts['cancelled_count']} cancelled; {counts['technical_failed_count']} technical errors; "
        f"{counts['terminal_count']} terminal; "
        f"{decisions.get('pass', 0)} PASS / {decisions.get('fail', 0)} FAIL / "
        f"{decisions.get('inconclusive', 0)} INCONCLUSIVE"
    )


def is_bootstrap_eta_sample(result: Mapping[str, Any]) -> bool:
    primary = result.get("primary") or {}
    return bool(
        result.get("status") == "completed"
        and result.get("technical_status", "completed") == "completed"
        and not result.get("cache_hit")
        and result.get("completion_reason") != "reused_result"
        and not primary.get("bootstrap_skipped_reason")
        and int(primary.get("bootstrap_repetitions") or 0) > 0
        and primary.get("ci_computed", True) is not False
    )


def replay_historical_cancellation(summary: Mapping[str, Any], evidence: Mapping[str, Any]) -> dict[str, Any]:
    """Separate read-only interpretation using bounded per-request evidence.

    A cancelled run or an exception name alone is deliberately insufficient.
    The audit must bind each original request hash/error to its observed time
    inside the independently established shutdown interval.
    """
    import copy
    out = copy.deepcopy(dict(summary))
    run_id = str(evidence.get("run_id") or "")
    bound_run_ids = {
        str(value) for row in summary.get("results") or []
        for value in (
            row.get("eval_run_id"), row.get("collection_eval_run_id"),
            (row.get("request_identity") or {}).get("eval_run_id"),
        ) if value
    }
    declared = str(summary.get("run_id") or summary.get("eval_run_id") or "")
    if declared:
        bound_run_ids.add(declared)
    if not run_id or bound_run_ids != {run_id}:
        raise ValueError("historical_cancel_run_binding_mismatch")
    if not evidence.get("source_evidence"):
        raise ValueError("historical_cancel_source_evidence_missing")
    def instant(value: Any) -> datetime:
        parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("historical_cancel_timestamp_timezone_missing")
        return parsed
    requested = instant(evidence.get("cancel_requested_at"))
    shutdown = instant(evidence.get("service_shutdown_at"))
    end = instant(evidence.get("shutdown_completed_at"))
    if not requested <= shutdown <= end:
        raise ValueError("historical_cancel_interval_invalid")
    events = evidence.get("events") or {}
    for result in out.get("results") or []:
        if result.get("status") == "completed":
            continue
        key = str(result.get("source_request") or "")
        event = events.get(key) or {}
        error = str(result.get("error") or "")
        name = error.split(":", 1)[0]
        if name not in {"CancelledError", "QualityServiceClosedError"}:
            continue
        sha = str(result.get("source_request_sha256") or "")
        if not sha or event.get("source_request_sha256") != sha or event.get("error") != error:
            continue
        if not event.get("observed_at") or not shutdown <= instant(event["observed_at"]) <= end:
            continue
        result["original_status"] = result.get("status")
        result["original_technical_status"] = result.get("technical_status")
        result.update({
            "status": "cancelled", "technical_status": "cancelled",
            "decision": "unavailable", "quality_decision": "not_evaluated",
            "completion_reason": "user_cancelled" if name == "CancelledError" else "service_closed_after_cancel",
            "legacy_cancel_projection": True,
        })
    out["original_status"] = summary.get("status")
    out["cancel_replay_evidence"] = copy.deepcopy(dict(evidence))
    out.update(summarize_requests(out.get("results") or []))
    out["failed_count"] = out["technical_failed_count"]
    if out["cancelled_count"]:
        out["status"] = "cancelled"
    return out
