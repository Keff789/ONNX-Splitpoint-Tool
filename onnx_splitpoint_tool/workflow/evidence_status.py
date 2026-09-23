"""Canonical Native runtime, semantic, claim and energy status axes."""
from __future__ import annotations

from typing import Any, Mapping, Sequence


SCHEMA = "onnx-splitpoint/native-evidence-status"
SCHEMA_VERSION = 5


_TASK_QUALITY_DECISIONS = {
    "reference_close": "reference_close", "accuracy_loss": "accuracy_loss", "not_estimable": "not_estimable",
    "pass": "pass",
    "passed": "pass",
    "fail": "fail",
    "failed": "fail",
    "inconclusive": "inconclusive",
    "reference": "reference",
}


def _task_quality_decision(
    row: Mapping[str, Any],
) -> tuple[str, str]:
    """Return the objective task-quality decision for one validation row.

    ``claim_ok`` is a structural compatibility alias and must not be used as
    the primary proof that task quality was evaluated.  Current rows expose a
    versioned task-quality decision directly.  The legacy ``claim_ok``
    fallback is retained only for archived rows that contain no task-quality
    decision field at all.

    Conflicting aliases fail closed as unavailable evidence.  A measured
    ``fail`` or ``inconclusive`` remains a complete scientific decision; only
    missing, pending, unavailable or conflicting evidence is incomplete.
    """
    values: list[tuple[str, Any]] = []
    direct_fields = (
        "task_quality_status",
        "task_quality_decision",
        "accuracy_gate_decision",
        "task_quality_gate_decision",
        "task_quality_gate_status",
    )
    for field in direct_fields:
        if field in row:
            values.append((field, row.get(field)))

    gate = row.get("task_quality_gate")
    if isinstance(gate, Mapping):
        for field in ("decision", "status"):
            if field in gate:
                values.append((f"task_quality_gate.{field}", gate.get(field)))

    axes = row.get("evidence_axes")
    task_axis = (
        axes.get("task_quality")
        if isinstance(axes, Mapping) else None
    )
    if isinstance(task_axis, Mapping):
        for field in ("decision", "status"):
            if field in task_axis:
                values.append((f"evidence_axes.task_quality.{field}", task_axis.get(field)))

    if not values:
        legacy = row.get("claim_ok")
        if isinstance(legacy, bool):
            return ("pass" if legacy else "fail", "legacy_claim_ok")
        return "unavailable", "task_quality_decision_missing"

    decisions: set[str] = set()
    unavailable_sources: list[str] = []
    for source, value in values:
        token = str(value or "").strip().lower()
        if source in {"task_quality_gate.status", "evidence_axes.task_quality.status"} and token in {"completed", "complete", "evaluated", "ok"}:
            # These are evaluation lifecycle states, not quality decisions.
            # An explicit fail/inconclusive result remains a completed result.
            continue
        decision = _TASK_QUALITY_DECISIONS.get(token)
        if decision:
            decisions.add(decision)
        elif token:
            unavailable_sources.append(source)
        else:
            unavailable_sources.append(source)

    if len(decisions) > 1:
        return "unavailable", "task_quality_decision_alias_conflict"
    if unavailable_sources and decisions:
        return "unavailable", "task_quality_decision_alias_conflict"
    if decisions:
        return next(iter(decisions)), "explicit_task_quality_decision"
    return "unavailable", "task_quality_decision_unavailable"


def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return default


def _optional_int(value: Any) -> int | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        return max(0, int(value))
    except (TypeError, ValueError, OverflowError):
        return None


def _rows(payload: Mapping[str, Any] | None) -> list[dict[str, Any]]:
    if not isinstance(payload, Mapping):
        return []
    return [
        dict(row) for row in list(payload.get("rows") or [])
        if isinstance(row, Mapping)
    ]


def _result_rows(
    payload: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]]:
    if isinstance(payload, Mapping):
        return _rows(payload)
    if isinstance(payload, Sequence) and not isinstance(
        payload, (str, bytes, bytearray)
    ):
        return [
            dict(row) for row in payload if isinstance(row, Mapping)
        ]
    return []


def _first_count(
    explicit: Any,
    counts: Mapping[str, Any],
    payload: Mapping[str, Any] | None,
    keys: Sequence[str],
) -> int | None:
    value = _optional_int(explicit)
    if value is not None:
        return value
    for source in (counts, payload or {}):
        for key in keys:
            value = _optional_int(source.get(key))
            if value is not None:
                return value
    return None


def _fraction(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, float(numerator) / float(denominator)), 4)


def _row_bool_count(
    rows: Sequence[Mapping[str, Any]],
    keys: Sequence[str],
    *,
    expected: bool,
) -> int:
    count = 0
    for row in rows:
        if not expected and (
            row.get("measurement_started") is False
            or str(row.get("skipped") or "").strip().lower()
            == "blocked_before_measurement"
            or str(row.get("status") or "").strip().lower()
            == "blocked_before_measurement"
        ):
            continue
        for key in keys:
            value = row.get(key)
            if isinstance(value, bool):
                count += int(value is expected)
                break
        else:
            token = str(row.get("measurement_status") or "").strip().lower()
            if expected and token in {
                "available", "complete", "completed", "ok", "passed",
                "success",
            }:
                count += 1
            elif not expected and token in {
                "failed", "measurement_failed", "error",
            }:
                count += 1
    return count


def _ledger_identity(row: Mapping[str, Any]) -> tuple[str, ...] | None:
    nested = row.get("row")
    source = nested if isinstance(nested, Mapping) else row
    from ..native_job_identity import native_backend, native_comparison
    backend = native_backend(source.get("backend"))
    model = str(
        source.get("model") or source.get("model_id") or ""
    ).strip()
    case = str(
        source.get("case") or source.get("case_id") or ""
    ).strip()
    setup = str(
        source.get("setup_id")
        or source.get("measurement_setup_id")
        or ""
    ).strip()
    comparison = str(
        source.get("comparison_backend") or ""
    ).strip().lower().replace("-", "_")
    comparison = {
        "hailo8_to_trt": "hailo8",
        "hailo8_to_tensorrt": "hailo8",
        "hailo10": "hailo10h",
        "hailo10h_to_trt": "hailo10h",
        "hailo10h_to_tensorrt": "hailo10h",
        "deepx_to_trt": "deepx",
        "deepx_to_tensorrt": "deepx",
    }.get(comparison, comparison)
    comparison = native_comparison(comparison)
    if not comparison:
        if backend in {"hailo8_to_trt", "native_full_hailo8"}:
            comparison = "hailo8"
        elif backend in {
            "hailo10h_to_trt", "native_full_hailo10",
            "native_full_hailo10h",
        }:
            comparison = "hailo10h"
        elif backend in {"deepx_to_trt", "native_full_deepx"}:
            comparison = "deepx"
    precision = (
        ""
        if backend.startswith("native_full_")
        else str(source.get("precision") or "").strip()
    )
    identity = (backend, model, case, setup, comparison, precision)
    required = identity[:5] if backend.startswith("native_full_") else identity
    return identity if all(required) else None


def _energy_accounting_projection(matrix, plan_rows, excluded_rows, result_rows, quality_rows=()):
    """Describe disjoint observed job sets without changing any admission gate."""
    expected_rows = [row for key in ("present_expected_rows", "successful_expected_rows", "failed_expected_rows", "missing_expected_rows")
                     for row in matrix.get(key, []) if isinstance(row, Mapping)]
    expected = {_ledger_identity(row) for row in expected_rows} - {None}
    result_keys = [_ledger_identity(row) for row in result_rows]
    excluded_keys = [_ledger_identity(row) for row in excluded_rows]
    results, excluded = set(result_keys) - {None}, set(excluded_keys) - {None}
    not_started = {_ledger_identity(row) for row in result_rows
                   if row.get("measurement_started") is False or
                   str(row.get("status") or row.get("skipped") or "") == "blocked_before_measurement"} - {None}
    conflicts = []
    for label, keys in (("results", result_keys), ("excluded", excluded_keys)):
        if None in keys:
            conflicts.append(label + "_identity_incomplete")
        if len(keys) != len(set(keys)):
            conflicts.append(label + "_duplicate_identity")
    if results & excluded:
        conflicts.append("measured_excluded_identity_overlap")
    # A conflicting identity is represented once in the display, while its
    # contradiction remains explicitly fatal to bookkeeping consistency.
    groups = {
        "measured": (results - not_started) & expected,
        "excluded": ((excluded - results) | not_started) & expected,
        "missing": expected - results - excluded,
        "unexpected": (results | excluded) - expected,
    }
    if groups["unexpected"]:
        conflicts.append("unexpected_identity")
    repeats, repeats_known = 0, True
    requested_repeats = collector_attempts = failed_collector_attempts = 0
    requested_known = attempts_known = bool(result_rows)
    per_row = []
    from ..native_energy_quality_admission import energy_quality_reason_projection
    quality_by_identity = {}
    for quality_row in quality_rows:
        key = _ledger_identity(quality_row)
        if key:
            decision, source_reason = _task_quality_decision(quality_row)
            quality_by_identity.setdefault(key, set()).add(decision)
    for item in result_rows:
        run = item.get("run") if isinstance(item.get("run"), Mapping) else item
        source = item.get("row") if isinstance(item.get("row"), Mapping) else item
        admission = source.get("energy_quality_admission")
        admission = admission if isinstance(admission, Mapping) else {}
        count = run.get("energy_aggregate_valid_repeat_count")
        if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
            repeats += count
        else:
            repeats_known = False
        requested = run.get("energy_aggregate_requested_repeat_count")
        if type(requested) is int and requested >= 0:
            requested_repeats += requested
        else:
            requested_known = False
        attempts = run.get("energy_collector_attempt_count")
        failed_attempts = run.get("energy_failed_collector_attempt_count")
        if type(attempts) is int and type(failed_attempts) is int and 0 <= failed_attempts <= attempts:
            collector_attempts += attempts
            failed_collector_attempts += failed_attempts
        else:
            attempts_known = False
        display_admission = dict(admission)
        decisions = quality_by_identity.get(_ledger_identity(item), set())
        if len(decisions) == 1 and next(iter(decisions)) in {"pass", "fail", "inconclusive", "reference", "reference_close", "accuracy_loss", "not_estimable"}:
            display_admission["local_task_quality_decision"] = next(iter(decisions))
        per_row.append({"identity": list(_ledger_identity(item)) if _ledger_identity(item) else None,
                        "local_quality_projection_source": "validation_row" if decisions else "stored_admission",
                        **energy_quality_reason_projection(display_admission)})
    return {
        "available": bool(expected_rows),
        "status": "invalid" if conflicts else "consistent" if expected_rows else "unavailable",
        "identity_conflicts": conflicts,
        "sets": {key: [list(identity) for identity in sorted(values)] for key, values in groups.items()},
        "counts": {key: len(values) for key, values in groups.items()},
        "expected_count": len(expected) if expected_rows else None,
        "valid_repetition_count": repeats if repeats_known else None,
        "requested_repetition_count": requested_repeats if requested_known else None,
        "collector_attempt_count": collector_attempts if attempts_known else None,
        "failed_collector_attempt_count": failed_collector_attempts if attempts_known else None,
        "row_quality": per_row,
        "campaign_comparison_released": False,  # caller supplies existing scientific gate
    }


def derive_native_evidence_status(
    *,
    run_mode: str,
    expected_matrix: Mapping[str, Any] | None,
    validation_payload: Mapping[str, Any] | None,
    validation_requested: bool,
    energy_requested: bool,
    energy_status: str = "",
    energy_strict: bool = False,
    energy_plan_payload: Mapping[str, Any] | None = None,
    energy_results_payload: (
        Mapping[str, Any] | Sequence[Mapping[str, Any]] | None
    ) = None,
    energy_counts: Mapping[str, Any] | None = None,
    energy_plan_included_count: int | None = None,
    energy_plan_excluded_count: int | None = None,
    energy_measurement_success_count: int | None = None,
    energy_measurement_failed_count: int | None = None,
    energy_claim_eligible_count: int | None = None,
    final_all_split_energy_required: bool | None = None,
) -> dict[str, Any]:
    """Derive four orthogonal axes from archived evidence.

    Claim eligibility is deliberately not treated as a runtime or semantic
    error.  A Smoke run can therefore be technically complete while all claim
    decisions are explicit ``False``.

    ``energy_plan_payload`` and ``energy_results_payload`` activate the v4
    coverage contract only when Native energy was requested.  When that
    contract is active, an empty plan is never complete and a successful
    planner/command status cannot stand in for measured matrix coverage.
    Disabled energy is an explicitly not-applicable axis: placeholder plan or
    result artifacts must not turn it into a failed coverage contract.  Calls
    that omit all v3 energy inputs retain the legacy status-token behaviour
    for archive compatibility.
    """
    mode = str(run_mode or "").strip().lower()
    diagnostic_only = mode == "smoke"
    strict_mode = mode in {"standard", "final"}
    from ..native_job_identity import project_known_build_exclusions, native_identity_key
    matrix = project_known_build_exclusions(expected_matrix or {})
    expected = _int(matrix.get("expected_row_count"))
    present = _int(matrix.get("present_expected_row_count"))
    successful = _int(matrix.get("successful_expected_row_count"))
    failed = _int(matrix.get("failed_expected_row_count"))
    missing = _int(matrix.get("missing_expected_row_count"))
    excluded = _int(matrix.get("excluded_expected_row_count"))
    if expected <= 0:
        expected = max(present + missing, successful + failed + missing)
    runtime_complete = bool(
        expected > 0
        and present == expected
        and successful + excluded == expected
        and failed == 0
        and missing == 0
        and not matrix.get("identity_unresolved_rows")
        and (not excluded or matrix.get("technical_execution_complete") is True)
    )
    runtime = {
        "status": ("complete_with_exclusions" if runtime_complete and excluded
                   else "complete" if runtime_complete else "incomplete"),
        "requested": True,
        "complete": runtime_complete,
        "expected_count": expected,
        "present_count": present,
        "successful_count": successful,
        "failed_count": failed,
        "missing_count": missing,
    }
    if excluded:
        runtime.update(excluded_count=excluded, executable_expected_count=max(0, expected - excluded),
                       all_requested_rows_measured=False)

    validation_rows = _rows(validation_payload)
    excluded_keys = {native_identity_key(row) for row in matrix.get("excluded_expected_rows", [])}
    validation_excluded_rows = [row for row in validation_rows if native_identity_key(row) in excluded_keys]
    validation_rows = [row for row in validation_rows if native_identity_key(row) not in excluded_keys]
    executable_expected = max(0, expected - excluded)
    row_count = len(validation_rows)
    semantic_available = sum(
        1 for row in validation_rows
        if row.get("semantic_available") is True
    )
    semantic_pass = sum(
        1 for row in validation_rows
        if row.get("semantic_available") is True
        and row.get("semantic_ok") is True
    )
    semantic_fail = sum(
        1 for row in validation_rows
        if row.get("semantic_available") is True
        and row.get("semantic_ok") is False
    )
    semantic_unavailable = max(0, row_count - semantic_available)
    semantic_complete = bool(
        validation_requested
        and expected > 0
        and row_count == executable_expected
        and semantic_available == executable_expected
    )
    semantic_all_pass = bool(
        semantic_complete and semantic_pass == executable_expected
        and semantic_fail == 0
    )
    semantics = {
        "status": (
            "complete_pass"
            if semantic_all_pass
            else "complete_fail"
            if semantic_complete
            else "incomplete"
            if validation_requested
            else "not_requested"
        ),
        "requested": bool(validation_requested),
        "complete": semantic_complete if validation_requested else True,
        "all_pass": semantic_all_pass if validation_requested else None,
        "row_count": row_count,
        "available_count": semantic_available,
        "pass_count": semantic_pass,
        "fail_count": semantic_fail,
        "unavailable_count": semantic_unavailable,
        "excluded_build_count": excluded,
        "excluded_validation_row_count": len(validation_excluded_rows),
        "executable_expected_count": executable_expected,
    }

    task_quality_rows = [
        _task_quality_decision(row) for row in validation_rows
    ]
    task_quality_counts = {
        status: sum(
            1 for decision, _reason in task_quality_rows
            if decision == status
        )
        for status in (
            "pass", "fail", "inconclusive", "reference", "reference_close", "accuracy_loss", "not_estimable", "unavailable",
        )
    }
    task_quality_conflict_count = sum(
        1 for _decision, reason in task_quality_rows
        if reason == "task_quality_decision_alias_conflict"
    )
    claim_decisions = sum(
        task_quality_counts[status]
        for status in ("pass", "fail", "inconclusive", "reference", "reference_close", "accuracy_loss", "not_estimable")
    )
    claim_eligible = sum(
        1
        for row, (decision, _reason) in zip(
            validation_rows, task_quality_rows,
        )
        if decision == "pass" and row.get("claim_ok") is True
    )
    claim_required = bool(strict_mode and validation_requested)
    claim_decisions_complete = bool(
        expected > 0
        and row_count == executable_expected
        and claim_decisions == executable_expected
    )
    # ``decision_complete`` retains the v2 requirement-aware meaning.  The
    # top-level ``claim_decisions_complete`` is the objective v3 axis.
    claim_complete = bool(not claim_required or claim_decisions_complete)
    claim_ready = bool(
        claim_required and claim_complete and claim_eligible > 0
    )
    claim = {
        "status": (
            "diagnostic_not_applicable"
            if diagnostic_only
            else "not_required"
            if not claim_required
            else "ready"
            if claim_ready
            else "complete_not_eligible"
            if claim_complete
            else "incomplete"
        ),
        "required": claim_required,
        "decision_complete": claim_complete,
        "decisions_complete": claim_decisions_complete,
        "decision_count": claim_decisions,
        "eligible_count": claim_eligible,
        "positive_claim_available": claim_eligible > 0,
    }
    task_quality = {
        "status": (
            "complete"
            if claim_decisions_complete
            else "incomplete"
            if validation_requested
            else "not_requested"
        ),
        "requested": bool(validation_requested),
        "complete": (
            claim_decisions_complete if validation_requested else True
        ),
        "row_count": row_count,
        "decision_count": claim_decisions,
        "pass_count": task_quality_counts["pass"],
        "fail_count": task_quality_counts["fail"],
        "inconclusive_count": task_quality_counts["inconclusive"],
        "reference_count": task_quality_counts["reference"],
        "unavailable_count": task_quality_counts["unavailable"],
        "alias_conflict_count": task_quality_conflict_count,
    }

    counts = (
        dict(energy_counts) if isinstance(energy_counts, Mapping) else {}
    )
    plan = (
        dict(energy_plan_payload)
        if isinstance(energy_plan_payload, Mapping) else None
    )
    results_mapping = (
        dict(energy_results_payload)
        if isinstance(energy_results_payload, Mapping) else None
    )
    result_rows = _result_rows(energy_results_payload)
    plan_rows = _rows(plan)
    excluded_rows = [
        row for row in list((plan or {}).get("excluded_rows") or [])
        if isinstance(row, Mapping)
    ]
    coverage_contract_active = bool(
        energy_requested
        and (
            energy_plan_payload is not None
            or energy_results_payload is not None
            or energy_counts is not None
            or energy_plan_included_count is not None
            or energy_plan_excluded_count is not None
            or energy_measurement_success_count is not None
            or energy_measurement_failed_count is not None
            or energy_claim_eligible_count is not None
        )
    )
    plan_included = _first_count(
        energy_plan_included_count,
        counts,
        plan,
        (
            "energy_plan_included_count",
            "plan_included_count",
            "included_count",
            "planned_row_count",
        ),
    )
    if plan_included is None:
        plan_included = len(plan_rows)
    plan_excluded = _first_count(
        energy_plan_excluded_count,
        counts,
        plan,
        (
            "energy_plan_excluded_count",
            "plan_excluded_count",
            "excluded_count",
        ),
    )
    if plan_excluded is None:
        plan_excluded = len(excluded_rows)
    measurement_success = _first_count(
        energy_measurement_success_count,
        counts,
        results_mapping,
        (
            "energy_measurement_success_count",
            "measurement_success_count",
            "successful_measurement_count",
            "success_count",
        ),
    )
    if measurement_success is None:
        measurement_success = _row_bool_count(
            result_rows,
            ("measurement_ok", "ok"),
            expected=True,
        )
    measurement_failed = _first_count(
        energy_measurement_failed_count,
        counts,
        results_mapping,
        (
            "energy_measurement_failed_count",
            "measurement_failed_count",
            "failed_measurement_count",
            "failure_count",
            "failed_count",
        ),
    )
    if measurement_failed is None:
        measurement_failed = _row_bool_count(
            result_rows,
            ("measurement_ok", "ok"),
            expected=False,
        )
    energy_claim_eligible = _first_count(
        energy_claim_eligible_count,
        counts,
        results_mapping,
        (
            "energy_claim_eligible_count",
            "claim_eligible_count",
            "eligible_count",
        ),
    )
    if energy_claim_eligible is None:
        energy_claim_eligible = _row_bool_count(
            result_rows,
            (
                "energy_claim_eligible",
                "eligible_for_scientific_claim",
                "claim_eligible",
            ),
            expected=True,
        )

    plan_completion_fraction = _fraction(
        measurement_success, plan_included
    )
    planned_matrix_coverage_fraction = _fraction(
        plan_included, expected
    )
    successful_matrix_coverage_fraction = _fraction(
        measurement_success, expected
    )
    plan_rows_declared = bool(
        plan is not None and "rows" in plan
    )
    excluded_rows_declared = bool(
        plan is not None and "excluded_rows" in plan
    )
    result_rows_declared = bool(
        (
            isinstance(energy_results_payload, Mapping)
            and "rows" in energy_results_payload
        )
        or (
            isinstance(energy_results_payload, Sequence)
            and not isinstance(
                energy_results_payload,
                (str, bytes, bytearray),
            )
        )
    )
    plan_identities = [
        _ledger_identity(row) for row in plan_rows
    ]
    excluded_identities = [
        _ledger_identity(row) for row in excluded_rows
    ]
    result_identities = [
        _ledger_identity(row) for row in result_rows
    ]
    expected_identity_row_keys = (
        "present_expected_rows",
        "successful_expected_rows",
        "failed_expected_rows",
        "missing_expected_rows",
    )
    expected_identity_rows_declared = any(
        key in matrix for key in expected_identity_row_keys
    )
    expected_identity_rows = [
        row
        for key in expected_identity_row_keys
        for row in list(matrix.get(key) or [])
        if isinstance(row, Mapping)
    ]
    expected_identities = [
        _ledger_identity(row) for row in expected_identity_rows
    ]
    plan_identity_set = {
        identity for identity in plan_identities
        if identity is not None
    }
    excluded_identity_set = {
        identity for identity in excluded_identities
        if identity is not None
    }
    result_identity_set = {
        identity for identity in result_identities
        if identity is not None
    }
    expected_identity_set = {
        identity for identity in expected_identities
        if identity is not None
    }
    expected_identity_contract_valid = bool(
        not expected_identity_rows_declared
        or (
            all(
                identity is not None
                for identity in expected_identities
            )
            and len(expected_identity_set) == expected
        )
    )
    expected_plan_identity_match = bool(
        not expected_identity_rows_declared
        or (
            expected_identity_contract_valid
            and (
                plan_identity_set | excluded_identity_set
            ) == expected_identity_set
        )
    )
    expected_result_identity_match = bool(
        not expected_identity_rows_declared
        or (
            expected_identity_contract_valid
            and result_identity_set <= expected_identity_set
        )
    )
    plan_identity_contract_valid = bool(
        not plan_rows_declared
        or (
            all(identity is not None for identity in plan_identities)
            and len(plan_identity_set) == len(plan_rows)
        )
    )
    excluded_identity_contract_valid = bool(
        not excluded_rows_declared
        or (
            all(
                identity is not None
                for identity in excluded_identities
            )
            and len(excluded_identity_set) == len(excluded_rows)
            and not (
                plan_identity_set & excluded_identity_set
            )
        )
    )
    result_identity_contract_valid = bool(
        not result_rows_declared
        or (
            all(identity is not None for identity in result_identities)
            and len(result_identity_set) == len(result_rows)
            and result_identity_set == plan_identity_set
        )
    )
    terminal_result_count = len(result_rows)
    not_started_preflight_count = sum(
        1
        for row in result_rows
        if (
            row.get("measurement_started") is False
            or str(row.get("skipped") or "").strip().lower()
            == "blocked_before_measurement"
            or str(row.get("status") or "").strip().lower()
            == "blocked_before_measurement"
        )
    )
    measurement_started_count = max(
        0,
        terminal_result_count - not_started_preflight_count,
    )
    preflight = (
        dict((plan or {}).get("preflight") or {})
        if isinstance((plan or {}).get("preflight"), Mapping)
        else {}
    )
    preflight_contract_declared = bool(
        plan is not None
        and (
            "preflight" in plan
            or "preflight_status" in plan
        )
    )
    preflight_contract_valid = bool(
        not preflight_contract_declared
        or (
            str(
                (plan or {}).get("preflight_status")
                or preflight.get("status")
                or ""
            ) == "passed"
            and str(preflight.get("status") or "") == "passed"
            and preflight.get("ok") is True
            and preflight.get("measurement_start_allowed") is True
            and preflight.get(
                "energy_plan_coverage_contract_valid"
            ) is True
        )
    )
    plan_ledger_valid = bool(
        (
            expected <= 0
            or plan_included + plan_excluded == expected
        )
        and (
            not plan_rows_declared
            or plan_included == len(plan_rows)
        )
        and (
            not excluded_rows_declared
            or plan_excluded == len(excluded_rows)
        )
        and plan_identity_contract_valid
        and excluded_identity_contract_valid
        and expected_identity_contract_valid
        and expected_plan_identity_match
        and preflight_contract_valid
    )
    observed_measurement_success = _row_bool_count(
        result_rows,
        ("measurement_ok", "ok"),
        expected=True,
    )
    observed_measurement_failed = _row_bool_count(
        result_rows,
        ("measurement_ok", "ok"),
        expected=False,
    )
    result_status_counts_consistent = bool(
        not result_rows_declared
        or (
            measurement_success == observed_measurement_success
            and measurement_failed == observed_measurement_failed
        )
    )
    result_ledger_valid = bool(
        measurement_success + measurement_failed
        == plan_included
        and (
            not result_rows_declared
            or measurement_success + measurement_failed
            == len(result_rows)
        )
        and result_identity_contract_valid
        and expected_result_identity_match
        and result_status_counts_consistent
    )
    terminal_result_ledger_valid = bool(
        (
            not result_rows_declared
            or terminal_result_count == plan_included
        )
        and result_identity_contract_valid
        and expected_result_identity_match
    )
    coverage_contract_valid = bool(
        plan_ledger_valid and result_ledger_valid
    )
    planned_measurements_complete = bool(
        plan_included > 0
        and measurement_success == plan_included
        and measurement_failed == 0
    )
    all_split_energy_complete = bool(
        energy_requested
        and coverage_contract_valid
        and expected > 0
        and plan_included == expected
        and plan_excluded == 0
        and measurement_success == expected
        and measurement_failed == 0
    )
    all_split_energy_required = bool(
        mode == "final" and energy_requested
        if final_all_split_energy_required is None
        else final_all_split_energy_required
    )

    energy_token = str(energy_status or "").strip().lower()
    if not energy_requested:
        energy_complete = True
        # A deliberately unrequested axis is complete by definition, but it is
        # not a failed or incomplete measurement.  Expose that distinction to
        # reports/UI explicitly instead of the ambiguous historical
        # ``disabled`` token.
        energy_axis_status = "not_applicable"
    elif coverage_contract_active:
        energy_complete = bool(
            planned_measurements_complete
            and coverage_contract_valid
        )
        energy_axis_status = (
            "complete"
            if energy_complete
            else "empty_plan"
            if plan_included == 0
            else "measurement_failed"
            if measurement_failed > 0
            else "incomplete"
        )
    else:
        energy_complete = energy_token in {
            "ok", "complete", "completed", "success", "passed",
        }
        energy_axis_status = (
            "complete" if energy_complete else "incomplete"
        )
    matrix_measurements_complete = bool(
        energy_complete
        if not coverage_contract_active
        else all_split_energy_complete
    ) if energy_requested else True

    # Preserve objective Energy gates above, but do not project their internal
    # zero/empty working values as negative evidence when the axis was never
    # requested.  ``None`` is the report-facing N/A value; the requirement-
    # aware completion fields remain True so disabled Energy cannot block an
    # otherwise complete run.  A contradictory explicit all-split requirement
    # remains visible and fail-closed.
    energy_not_applicable = not bool(energy_requested)

    def energy_evidence_value(value: Any) -> Any:
        return None if energy_not_applicable else value

    coverage_contract_status = (
        "not_applicable"
        if energy_not_applicable
        else "valid"
        if coverage_contract_active and coverage_contract_valid
        else "invalid"
        if coverage_contract_active
        else "inactive"
    )
    reported_all_split_energy_complete = (
        None
        if energy_not_applicable and not all_split_energy_required
        else all_split_energy_complete
    )
    accounting = _energy_accounting_projection(matrix, plan_rows, excluded_rows, result_rows, validation_rows)
    energy = {
        "accounting": accounting if energy_requested else None,
        "status": energy_axis_status,
        "requested": bool(energy_requested),
        "strict": bool(energy_requested and energy_strict),
        "coverage_contract_active": coverage_contract_active,
        "coverage_contract_status": coverage_contract_status,
        "coverage_source": (
            "not_applicable"
            if energy_not_applicable else
            "v4_identity_plan_results_counts"
            if coverage_contract_active else "legacy_status_token"
        ),
        "coverage_contract_valid": energy_evidence_value(
            coverage_contract_valid
        ),
        "preflight_contract_valid": energy_evidence_value(
            preflight_contract_valid
        ),
        "plan_ledger_valid": energy_evidence_value(plan_ledger_valid),
        "result_ledger_valid": energy_evidence_value(result_ledger_valid),
        "terminal_result_ledger_valid": (
            energy_evidence_value(terminal_result_ledger_valid)
        ),
        "plan_identity_contract_valid": (
            energy_evidence_value(plan_identity_contract_valid)
        ),
        "excluded_identity_contract_valid": (
            energy_evidence_value(excluded_identity_contract_valid)
        ),
        "result_identity_contract_valid": (
            energy_evidence_value(result_identity_contract_valid)
        ),
        "expected_identity_contract_valid": (
            energy_evidence_value(expected_identity_contract_valid)
        ),
        "expected_plan_identity_match": (
            energy_evidence_value(expected_plan_identity_match)
        ),
        "expected_result_identity_match": (
            energy_evidence_value(expected_result_identity_match)
        ),
        "result_status_counts_consistent": (
            energy_evidence_value(result_status_counts_consistent)
        ),
        "matrix_expected_count": energy_evidence_value(expected),
        "plan_included_count": energy_evidence_value(plan_included),
        "plan_excluded_count": energy_evidence_value(plan_excluded),
        "measurement_success_count": energy_evidence_value(
            measurement_success
        ),
        "measurement_failed_count": energy_evidence_value(
            measurement_failed
        ),
        "terminal_result_count": energy_evidence_value(
            terminal_result_count
        ),
        "not_started_preflight_count": (
            energy_evidence_value(not_started_preflight_count)
        ),
        "measurement_started_count": energy_evidence_value(
            measurement_started_count
        ),
        "claim_eligible_count": energy_evidence_value(
            energy_claim_eligible
        ),
        "planned_completion_fraction": energy_evidence_value(
            plan_completion_fraction
        ),
        "plan_completion_fraction": energy_evidence_value(
            plan_completion_fraction
        ),
        "planned_matrix_coverage_fraction": (
            energy_evidence_value(planned_matrix_coverage_fraction)
        ),
        "successful_matrix_coverage_fraction": (
            energy_evidence_value(successful_matrix_coverage_fraction)
        ),
        "matrix_coverage_fraction": (
            energy_evidence_value(successful_matrix_coverage_fraction)
        ),
        "planned_measurements_complete": (
            energy_complete
            if not energy_requested or not coverage_contract_active
            else planned_measurements_complete
        ),
        "matrix_measurements_complete": matrix_measurements_complete,
        "final_all_split_required": all_split_energy_required,
        "final_all_split_complete": reported_all_split_energy_complete,
    }

    declared_technical_errors = _int(
        (validation_payload or {}).get("technical_error_count")
        if isinstance(validation_payload, Mapping) else 0
    )
    validation_chain_complete = bool(
        isinstance(validation_payload, Mapping)
        and validation_payload
        and (validation_payload or {}).get("empty_output_error") is not True
        and (validation_payload or {}).get(
            "technical_chain_complete"
        ) is not False
        and declared_technical_errors == 0
    )
    technical_quality_failure = bool(
        not runtime_complete
        or (
            validation_requested
            and not validation_chain_complete
        )
        or (
            energy_requested
            and energy_strict
            and not energy_complete
        )
    )
    evidence_complete = bool(
        runtime_complete
        and (not validation_requested or semantic_complete)
        and (not validation_requested or claim_decisions_complete)
        and matrix_measurements_complete
        and not technical_quality_failure
    )
    scientific_energy_ready = bool(
        (not energy_requested and not all_split_energy_required)
        or (
            matrix_measurements_complete
            and (
                not all_split_energy_required
                or all_split_energy_complete
            )
        )
    )
    scientific_ready = bool(
        evidence_complete
        and excluded == 0
        and (not validation_requested or semantic_complete)
        and (not claim_required or claim_decisions_complete)
        and scientific_energy_ready
    )
    technical_complete = bool(
        evidence_complete
        or (excluded > 0 and runtime_complete
            and (not validation_requested or (semantic_complete and claim_decisions_complete))
            and (not energy_requested or energy_complete)
            and not technical_quality_failure)
    )
    technical_status = (
        "partial"
        if technical_quality_failure
        else "complete"
        if technical_complete
        else "incomplete"
    )
    scientific_status = "ready" if scientific_ready else "not_ready"
    accounting["campaign_comparison_released"] = bool(scientific_ready and energy_claim_eligible > 0)
    accounting["summary"] = (
        f"Energy measurement: {measurement_success}/{plan_included} planned rows successful; "
        f"valid repetitions={accounting['valid_repetition_count'] if accounting['valid_repetition_count'] is not None else 'unavailable'}. "
        f"Original matrix: {measurement_success}/{expected} measured successfully, {plan_excluded} excluded. "
        f"Campaign comparison: {'released' if accounting['campaign_comparison_released'] else 'not released'}. "
        "Local task quality and endpoint reasons remain separate."
    )
    return {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "run_mode": mode,
        "diagnostic_only": diagnostic_only,
        "runtime": runtime,
        "semantics": semantics,
        "task_quality": task_quality,
        "claim": claim,
        "energy": energy,
        "technical_quality_failure": technical_quality_failure,
        "evidence_complete": evidence_complete,
        "scientific_ready": scientific_ready,
        "technical_status": technical_status,
        "technical_complete": technical_complete,
        "known_build_excluded_count": excluded,
        "validation_technical_error_count": declared_technical_errors,
        "validation_measured_technical_error_count": _optional_int((validation_payload or {}).get("measured_technical_error_count")),
        "validation_empty_output_error": bool((validation_payload or {}).get("empty_output_error")),
        "claim_decisions_complete": claim_decisions_complete,
        "scientific_status": scientific_status,
        "positive_performance_claim_available": claim_eligible > 0,
        "positive_energy_claim_available": energy_evidence_value(
            energy_claim_eligible > 0
        ),
        "energy_matrix_expected_count": energy_evidence_value(expected),
        "energy_plan_included_count": energy_evidence_value(plan_included),
        "energy_plan_excluded_count": energy_evidence_value(plan_excluded),
        "energy_measurement_success_count": energy_evidence_value(
            measurement_success
        ),
        "energy_measurement_failed_count": energy_evidence_value(
            measurement_failed
        ),
        "energy_terminal_result_count": energy_evidence_value(
            terminal_result_count
        ),
        "energy_not_started_preflight_count": (
            energy_evidence_value(not_started_preflight_count)
        ),
        "energy_measurement_started_count": energy_evidence_value(
            measurement_started_count
        ),
        "energy_claim_eligible_count": energy_evidence_value(
            energy_claim_eligible
        ),
        "energy_planned_completion_fraction": (
            energy_evidence_value(plan_completion_fraction)
        ),
        "energy_plan_completion_fraction": energy_evidence_value(
            plan_completion_fraction
        ),
        "energy_planned_matrix_coverage_fraction": (
            energy_evidence_value(planned_matrix_coverage_fraction)
        ),
        "energy_successful_matrix_coverage_fraction": (
            energy_evidence_value(successful_matrix_coverage_fraction)
        ),
        "energy_matrix_coverage_fraction": (
            energy_evidence_value(successful_matrix_coverage_fraction)
        ),
        "energy_coverage_contract_active": coverage_contract_active,
        "energy_coverage_contract_status": coverage_contract_status,
        "energy_coverage_contract_valid": (
            energy_evidence_value(coverage_contract_valid)
        ),
        "energy_plan_ledger_valid": energy_evidence_value(
            plan_ledger_valid
        ),
        "energy_result_ledger_valid": energy_evidence_value(
            result_ledger_valid
        ),
        "energy_expected_identity_contract_valid": (
            energy_evidence_value(expected_identity_contract_valid)
        ),
        "energy_expected_plan_identity_match": (
            energy_evidence_value(expected_plan_identity_match)
        ),
        "energy_expected_result_identity_match": (
            energy_evidence_value(expected_result_identity_match)
        ),
        "energy_terminal_result_ledger_valid": (
            energy_evidence_value(terminal_result_ledger_valid)
        ),
        "final_all_split_energy_required": all_split_energy_required,
        "final_all_split_energy_complete": (
            reported_all_split_energy_complete
        ),
        "deprecated_aliases": {
            "claim_ready": {
                "deprecated": True,
                "value": claim_ready,
                "replacement": (
                    "claim_decisions_complete plus explicit scientific "
                    "eligibility"
                ),
            },
            "claim.ready": {
                "deprecated": True,
                "value": claim_ready,
                "replacement": (
                    "claim_decisions_complete plus explicit scientific "
                    "eligibility"
                ),
            },
            "energy_complete": {
                "deprecated": True,
                "value": energy_complete,
                "replacement": (
                    "energy_plan_completion_fraction, "
                    "energy_planned_matrix_coverage_fraction, "
                    "energy_successful_matrix_coverage_fraction and "
                    "energy_coverage_contract_valid"
                ),
            },
            "energy.complete": {
                "deprecated": True,
                "value": energy_complete,
                "replacement": (
                    "energy_plan_completion_fraction, "
                    "energy_planned_matrix_coverage_fraction, "
                    "energy_successful_matrix_coverage_fraction and "
                    "energy_coverage_contract_valid"
                ),
            },
        },
    }


def project_native_evidence_status(
    evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the stable report-facing projection of Native evidence.

    Report writers must not independently reinterpret the legacy
    ``claim.decision_complete`` or ``energy.complete`` aliases.  In
    particular, ``claim.decision_complete`` is requirement-aware and can be
    true in diagnostic modes even when objective decisions are missing.
    This projection therefore consumes only the v3 objective axis
    (top-level ``claim_decisions_complete`` or
    ``claim.decisions_complete``).

    Missing evidence stays explicitly unavailable.  ``None`` values are used
    instead of false/zero values so an absent artifact cannot look like a
    measured negative result or an empty completed plan.
    """
    source = dict(evidence) if isinstance(evidence, Mapping) else {}
    available = bool(source)
    energy = (
        dict(source.get("energy") or {})
        if isinstance(source.get("energy"), Mapping) else {}
    )
    claim = (
        dict(source.get("claim") or {})
        if isinstance(source.get("claim"), Mapping) else {}
    )
    task_quality = (
        dict(source.get("task_quality") or {})
        if isinstance(source.get("task_quality"), Mapping) else {}
    )

    def objective_bool(value: Any) -> bool | None:
        return value if isinstance(value, bool) else None

    energy_requested = objective_bool(energy.get("requested"))
    energy_not_applicable = bool(
        available and energy_requested is False
    )
    coverage_contract_active = objective_bool(
        source.get("energy_coverage_contract_active")
    )
    if coverage_contract_active is None:
        coverage_contract_active = objective_bool(
            energy.get("coverage_contract_active")
        )
    coverage_contract_status = str(
        source.get("energy_coverage_contract_status")
        or energy.get("coverage_contract_status")
        or ""
    ).strip()
    coverage_source = str(
        energy.get("coverage_source") or ""
    ).strip()

    def count(top_level_key: str, nested_key: str) -> int | None:
        value = _optional_int(source.get(top_level_key))
        if value is None:
            value = _optional_int(energy.get(nested_key))
        return value

    def fraction(
        top_level_keys: Sequence[str],
        nested_keys: Sequence[str],
        numerator: int | None,
        denominator: int | None,
    ) -> float | None:
        for container, keys in (
            (source, top_level_keys),
            (energy, nested_keys),
        ):
            for key in keys:
                value = container.get(key)
                if value is not None and not isinstance(value, bool):
                    try:
                        return float(value)
                    except (TypeError, ValueError, OverflowError):
                        continue
        if numerator is None or denominator is None:
            return None
        return _fraction(numerator, denominator)

    technical_status = (
        str(source.get("technical_status") or "unavailable")
        if available else "unavailable"
    )
    technical_complete = objective_bool(source.get("technical_complete"))
    if technical_complete is None:
        technical_complete = objective_bool(source.get("evidence_complete"))
    if technical_complete is None and available:
        if technical_status == "complete":
            technical_complete = True
        elif technical_status in {"failed", "incomplete"}:
            technical_complete = False

    claim_decisions_complete = objective_bool(
        source.get("claim_decisions_complete")
    )
    if claim_decisions_complete is None:
        claim_decisions_complete = objective_bool(
            claim.get("decisions_complete")
        )

    scientific_ready = objective_bool(source.get("scientific_ready"))
    scientific_status = (
        str(source.get("scientific_status") or "").strip()
        if available else ""
    )
    if not scientific_status:
        scientific_status = (
            "ready"
            if scientific_ready is True
            else "not_ready"
            if scientific_ready is False
            else "unavailable"
        )

    matrix_denominator = count(
        "energy_matrix_expected_count", "matrix_expected_count"
    )
    plan_denominator = count(
        "energy_plan_included_count", "plan_included_count"
    )
    plan_excluded = count(
        "energy_plan_excluded_count", "plan_excluded_count"
    )
    measurement_success = count(
        "energy_measurement_success_count", "measurement_success_count"
    )
    measurement_failed = count(
        "energy_measurement_failed_count", "measurement_failed_count"
    )
    measurement_started = count(
        "energy_measurement_started_count", "measurement_started_count"
    )
    not_started_preflight = count(
        "energy_not_started_preflight_count",
        "not_started_preflight_count",
    )
    terminal_result_count = count(
        "energy_terminal_result_count", "terminal_result_count"
    )
    if (
        measurement_started is None
        and terminal_result_count is not None
        and not_started_preflight is not None
    ):
        measurement_started = max(
            0, terminal_result_count - not_started_preflight,
        )
    if (
        not_started_preflight is None
        and terminal_result_count is not None
        and measurement_started is not None
    ):
        not_started_preflight = max(
            0, terminal_result_count - measurement_started,
        )
    if (
        terminal_result_count is None
        and measurement_started is not None
        and not_started_preflight is not None
    ):
        terminal_result_count = (
            measurement_started + not_started_preflight
        )

    # A terminal ``blocked_before_measurement`` row is still part of the
    # admitted plan.  Older producers could archive those terminal rows while
    # leaving ``plan_included_count`` at zero, which made every downstream
    # report render the blocked plan as a misleading 0/0 completion.  Recover
    # only the lower bound proven by the terminal/start ledger; never infer a
    # plan from an absent evidence artifact.
    activity_plan_count = (
        measurement_started + not_started_preflight
        if (
            measurement_started is not None
            and not_started_preflight is not None
        )
        else terminal_result_count
    )
    if activity_plan_count is not None:
        plan_denominator = max(
            int(plan_denominator or 0), activity_plan_count,
        )

    # ``measurement_failed`` denotes attempts that actually entered the
    # measurement window.  Preflight-blocked terminal rows are represented by
    # ``not_started_preflight`` and must not also look like failed attempts.
    if measurement_started is not None and measurement_failed is not None:
        measured_success = int(measurement_success or 0)
        measurement_failed = min(
            measurement_failed,
            max(0, measurement_started - measured_success),
        )
    claim_eligible = count(
        "energy_claim_eligible_count", "claim_eligible_count"
    )
    zero_start_blocked = bool(
        plan_denominator is not None
        and plan_denominator > 0
        and measurement_started == 0
        and not_started_preflight is not None
        and not_started_preflight >= plan_denominator
    )
    plan_fraction = fraction(
        (
            "energy_plan_completion_fraction",
            "energy_planned_completion_fraction",
        ),
        (
            "plan_completion_fraction",
            "planned_completion_fraction",
        ),
        measurement_success,
        plan_denominator,
    )
    planned_matrix_fraction = fraction(
        ("energy_planned_matrix_coverage_fraction",),
        ("planned_matrix_coverage_fraction",),
        plan_denominator,
        matrix_denominator,
    )
    successful_matrix_fraction = fraction(
        (
            "energy_successful_matrix_coverage_fraction",
            "energy_matrix_coverage_fraction",
        ),
        (
            "successful_matrix_coverage_fraction",
            "matrix_coverage_fraction",
        ),
        measurement_success,
        matrix_denominator,
    )
    plan_complete = objective_bool(
        energy.get("planned_measurements_complete")
    )
    final_all_split_required = objective_bool(
        source.get("final_all_split_energy_required")
    )
    final_all_split_complete = objective_bool(
        source.get("final_all_split_energy_complete")
    )
    coverage_contract_valid = objective_bool(
        source.get("energy_coverage_contract_valid")
    )
    if coverage_contract_valid is None:
        coverage_contract_valid = objective_bool(
            energy.get("coverage_contract_valid")
        )
    plan_ledger_valid = objective_bool(
        source.get("energy_plan_ledger_valid")
    )
    if plan_ledger_valid is None:
        plan_ledger_valid = objective_bool(
            energy.get("plan_ledger_valid")
        )
    result_ledger_valid = objective_bool(
        source.get("energy_result_ledger_valid")
    )
    if result_ledger_valid is None:
        result_ledger_valid = objective_bool(
            energy.get("result_ledger_valid")
        )
    preflight_contract_valid = objective_bool(
        energy.get("preflight_contract_valid")
    )
    terminal_result_ledger_valid = objective_bool(
        source.get("energy_terminal_result_ledger_valid")
    )
    if terminal_result_ledger_valid is None:
        terminal_result_ledger_valid = objective_bool(
            energy.get("terminal_result_ledger_valid")
        )
    expected_identity_contract_valid = objective_bool(
        source.get("energy_expected_identity_contract_valid")
    )
    if expected_identity_contract_valid is None:
        expected_identity_contract_valid = objective_bool(
            energy.get("expected_identity_contract_valid")
        )
    expected_plan_identity_match = objective_bool(
        source.get("energy_expected_plan_identity_match")
    )
    if expected_plan_identity_match is None:
        expected_plan_identity_match = objective_bool(
            energy.get("expected_plan_identity_match")
        )
    expected_result_identity_match = objective_bool(
        source.get("energy_expected_result_identity_match")
    )
    if expected_result_identity_match is None:
        expected_result_identity_match = objective_bool(
            energy.get("expected_result_identity_match")
        )
    matrix_measurements_complete = objective_bool(
        energy.get("matrix_measurements_complete")
    )
    positive_performance_claim_available = objective_bool(
        source.get("positive_performance_claim_available")
    )
    if positive_performance_claim_available is None:
        positive_performance_claim_available = objective_bool(
            claim.get("positive_claim_available")
        )
    positive_energy_claim_available = objective_bool(
        source.get("positive_energy_claim_available")
    )
    energy_status = (
        str(energy.get("status") or "").strip()
        if available else ""
    )
    if energy_not_applicable:
        # Normalize both current evidence and archived pre-fix evidence.  Old
        # archives could contain zero-count placeholder ledgers with
        # ``requested=false``; reports must not reinterpret those placeholders
        # as a measured 0/N coverage failure.
        energy_status = "not_applicable"
        coverage_contract_active = False
        coverage_contract_status = "not_applicable"
        coverage_source = "not_applicable"
        matrix_denominator = None
        plan_denominator = None
        plan_excluded = None
        measurement_success = None
        measurement_failed = None
        measurement_started = None
        not_started_preflight = None
        terminal_result_count = None
        claim_eligible = None
        plan_fraction = None
        planned_matrix_fraction = None
        successful_matrix_fraction = None
        plan_complete = None
        coverage_contract_valid = None
        plan_ledger_valid = None
        result_ledger_valid = None
        preflight_contract_valid = None
        terminal_result_ledger_valid = None
        expected_identity_contract_valid = None
        expected_plan_identity_match = None
        expected_result_identity_match = None
        matrix_measurements_complete = None
        positive_energy_claim_available = None
        if final_all_split_required is not True:
            final_all_split_complete = None
        zero_start_blocked = False
    if zero_start_blocked:
        # Counts are the objective execution evidence.  They fail closed over
        # stale ``complete`` aliases left by a parent stage that returned
        # before any measurement started.
        energy_status = "blocked_before_measurement"
        measurement_success = 0
        measurement_failed = 0
        claim_eligible = 0
        plan_fraction = _fraction(0, plan_denominator)
        planned_matrix_fraction = (
            _fraction(plan_denominator, matrix_denominator)
            if matrix_denominator is not None else None
        )
        successful_matrix_fraction = (
            _fraction(0, matrix_denominator)
            if matrix_denominator is not None else None
        )
        plan_complete = False
        matrix_measurements_complete = False
        final_all_split_complete = False
        positive_energy_claim_available = False
        technical_status = "failed"
        technical_complete = False
        scientific_status = "not_ready"
        scientific_ready = False
    elif not energy_status:
        energy_status = "unavailable"

    def task_quality_count(key: str) -> int | None:
        return _optional_int(task_quality.get(key))

    return {
        "energy_accounting": energy.get("accounting") if available else None,
        "energy_accounting_summary": (energy.get("accounting") or {}).get("summary") if available else None,
        "schema": "onnx-splitpoint/native-evidence-report-projection",
        "schema_version": 3,
        "available": available,
        "source_schema": source.get("schema") if available else None,
        "source_schema_version": (
            _optional_int(source.get("schema_version"))
            if available else None
        ),
        "run_mode": (
            str(source.get("run_mode") or "").strip() or None
            if available else None
        ),
        "technical_status": technical_status,
        "technical_complete": technical_complete,
        "claim_decisions_complete": claim_decisions_complete,
        "runtime_status": (
            str((source.get("runtime") or {}).get("status") or "unavailable")
            if isinstance(source.get("runtime"), Mapping)
            else "unavailable"
        ),
        "semantic_status": (
            str(
                (source.get("semantics") or {}).get("status")
                or "unavailable"
            )
            if isinstance(source.get("semantics"), Mapping)
            else "unavailable"
        ),
        "claim_status": (
            str((source.get("claim") or {}).get("status") or "unavailable")
            if isinstance(source.get("claim"), Mapping)
            else "unavailable"
        ),
        "task_quality_status": (
            str(task_quality.get("status") or "unavailable")
            if available else "unavailable"
        ),
        "task_quality_decision_count": task_quality_count(
            "decision_count"
        ),
        "task_quality_pass_count": task_quality_count("pass_count"),
        "task_quality_fail_count": task_quality_count("fail_count"),
        "task_quality_inconclusive_count": task_quality_count(
            "inconclusive_count"
        ),
        "task_quality_unavailable_count": task_quality_count(
            "unavailable_count"
        ),
        "positive_performance_claim_available": (
            positive_performance_claim_available
        ),
        "positive_energy_claim_available": (
            positive_energy_claim_available
        ),
        "scientific_status": scientific_status,
        "scientific_ready": scientific_ready,
        "energy_status": energy_status,
        "energy_execution_status": energy_status,
        "energy_requested": energy_requested,
        "energy_coverage_contract_active": coverage_contract_active,
        "energy_coverage_contract_status": coverage_contract_status or (
            "valid"
            if coverage_contract_active is True
            and coverage_contract_valid is True
            else "invalid"
            if coverage_contract_active is True
            else "inactive"
            if coverage_contract_active is False
            else "unavailable"
        ),
        "energy_coverage_source": coverage_source or None,
        "energy_zero_start_blocked": zero_start_blocked,
        "energy_measurement_success_count": measurement_success,
        "energy_measurement_failed_count": measurement_failed,
        "energy_measurement_started_count": measurement_started,
        "energy_measurement_attempted_count": measurement_started,
        "energy_not_started_preflight_count": not_started_preflight,
        "energy_terminal_result_count": terminal_result_count,
        "energy_claim_eligible_count": claim_eligible,
        "energy_plan_denominator_count": plan_denominator,
        "energy_plan_included_count": plan_denominator,
        "energy_plan_excluded_count": plan_excluded,
        "energy_matrix_denominator_count": matrix_denominator,
        "energy_matrix_expected_count": matrix_denominator,
        "energy_plan_completion_fraction": plan_fraction,
        "energy_planned_completion_fraction": plan_fraction,
        "energy_planned_matrix_coverage_fraction": (
            planned_matrix_fraction
        ),
        "energy_successful_matrix_coverage_fraction": (
            successful_matrix_fraction
        ),
        "energy_matrix_coverage_fraction": (
            successful_matrix_fraction
        ),
        "energy_coverage_contract_valid": coverage_contract_valid,
        "energy_plan_ledger_valid": plan_ledger_valid,
        "energy_result_ledger_valid": result_ledger_valid,
        "energy_preflight_contract_valid": preflight_contract_valid,
        "energy_terminal_result_ledger_valid": (
            terminal_result_ledger_valid
        ),
        "energy_expected_identity_contract_valid": (
            expected_identity_contract_valid
        ),
        "energy_expected_plan_identity_match": (
            expected_plan_identity_match
        ),
        "energy_expected_result_identity_match": (
            expected_result_identity_match
        ),
        "energy_matrix_measurements_complete": (
            matrix_measurements_complete
        ),
        "energy_plan_completion": {
            "numerator_count": measurement_success,
            "denominator_count": plan_denominator,
            "excluded_count": plan_excluded,
            "fraction": plan_fraction,
            "complete": plan_complete,
        },
        "energy_measurement_attempts": {
            "attempted_count": measurement_started,
            "not_started_preflight_count": not_started_preflight,
            "terminal_result_count": terminal_result_count,
            "planned_count": plan_denominator,
            "status": energy_status,
            "complete": (
                False if zero_start_blocked else plan_complete
            ),
        },
        "energy_planned_matrix_coverage": {
            "numerator_count": plan_denominator,
            "denominator_count": matrix_denominator,
            "fraction": planned_matrix_fraction,
            "complete": (
                plan_denominator == matrix_denominator
                if (
                    plan_denominator is not None
                    and matrix_denominator is not None
                )
                else None
            ),
        },
        "energy_successful_matrix_coverage": {
            "numerator_count": measurement_success,
            "denominator_count": matrix_denominator,
            "fraction": successful_matrix_fraction,
            "complete": final_all_split_complete,
        },
        "energy_matrix_coverage": {
            "numerator_count": measurement_success,
            "denominator_count": matrix_denominator,
            "fraction": successful_matrix_fraction,
            "complete": final_all_split_complete,
        },
        "final_all_split_energy_required": final_all_split_required,
        "final_all_split_energy_complete": final_all_split_complete,
    }


def blocking_status(
    evidence: Mapping[str, Any] | None,
    *,
    run_mode: str,
) -> str:
    """Return the workflow downgrade required by Native evidence."""
    if not isinstance(evidence, Mapping) or not evidence:
        return ""
    mode = str(run_mode or evidence.get("run_mode") or "").strip().lower()
    if evidence.get("global_integrity_failure") is True or evidence.get("publication_failed") is True:
        return "failed"
    if evidence.get("technical_quality_failure") is True:
        return "partial"
    semantics = (
        evidence.get("semantics")
        if isinstance(evidence.get("semantics"), Mapping) else {}
    )
    if (
        semantics.get("requested") is True
        and semantics.get("complete") is not True
    ):
        # Missing or deliberately unavailable scientific evidence is not a
        # runtime/technical failure.  Standard is a development-screening
        # mode. Missing evidence remains an explicit partial result in Final too;
        # claim readiness remains closed until every decision is available.
        return "partial"
    claim = (
        evidence.get("claim")
        if isinstance(evidence.get("claim"), Mapping) else {}
    )
    if (
        claim.get("required") is True
        and evidence.get("claim_decisions_complete") is not True
    ):
        # A complete fail or inconclusive task-quality result is valid negative
        # evidence.  Only a missing/unavailable decision downgrades the run;
        # the absence of a positive claim candidate does not.
        return "partial"
    energy = (
        evidence.get("energy")
        if isinstance(evidence.get("energy"), Mapping) else {}
    )
    if (
        evidence.get("final_all_split_energy_required") is True
        and evidence.get("final_all_split_energy_complete") is not True
    ):
        return "partial"
    if (evidence.get("known_build_excluded_count", 0) > 0
            and evidence.get("technical_complete") is True
            and (energy.get("requested") is not True
                 or (energy.get("planned_measurements_complete") is True
                     and energy.get("coverage_contract_valid") is True))):
        # Exact build exclusions are terminal scientific outcomes. The original
        # matrix coverage and scientific-ready fields remain negative, but they
        # do not create a fictitious failed measurement or pending runtime job.
        return ""
    if (
        energy.get("requested") is True
        and energy.get("coverage_contract_active") is True
        and energy.get("matrix_measurements_complete") is not True
    ):
        return "partial"
    if (
        energy.get("requested") is True
        and (
            energy.get("planned_measurements_complete") is not True
            or energy.get("coverage_contract_valid") is not True
        )
    ):
        return "partial"
    return ""


def workflow_completion_projection(
    technical_status: str,
    *,
    native_evidence: Mapping[str, Any] | None = None,
    central_quality: Mapping[str, Any] | None = None,
    other_warning_count: int = 0,
    generic_excluded_count: int = 0,
) -> dict[str, Any]:
    """One display projection; never changes a measurement or quality decision.

    Runtime, validation and energy denominators overlap. In particular a dump
    conflict is a validation error on an existing measured row, not a new
    failed/missing runtime row. A missing counter remains unavailable.
    """
    evidence = native_evidence if isinstance(native_evidence, Mapping) else {}
    quality = central_quality if isinstance(central_quality, Mapping) else {}
    runtime = evidence.get("runtime") or {}
    energy = evidence.get("energy") or {}
    task_quality = evidence.get("task_quality") or {}
    decisions = quality.get("decision_counts") or quality.get("quality_decision_counts") or {}
    from ..quality_lifecycle import summarize_requests
    uncertainty = (summarize_requests(quality["results"])["quality_uncertainty_counts"]
                   if "results" in quality else quality.get("quality_uncertainty_counts") or {})
    status = str(technical_status or "unavailable").strip().lower()
    if status in {"completed", "success", "complete"}:
        status = "ok"
    def count(source: Mapping[str, Any], key: str) -> int | None:
        return _optional_int(source.get(key)) if key in source else None
    counts = {
        "native_selected": count(runtime, "expected_count"),
        "native_present": count(runtime, "present_count"),
        "native_measured": count(runtime, "successful_count"),
        "native_excluded": _int(evidence.get("known_build_excluded_count")),
        "native_failed_or_blocked": count(runtime, "failed_count"),
        "native_missing": count(runtime, "missing_count"),
        "native_validation_technical_errors": count(evidence, "validation_measured_technical_error_count"),
        "native_validation_total_technical_errors": count(evidence, "validation_technical_error_count"),
        "central_quality_started": count(quality, "request_count"),
        "central_quality_completed": count(quality, "completed_count"),
        "central_quality_cancelled": count(quality, "cancelled_count"),
        "central_quality_technical_failed": count(quality, "technical_failed_count"),
        "central_quality_accuracy_loss": _int(decisions.get("accuracy_loss")),
        "central_quality_reference_close": _int(decisions.get("reference_close")),
        "central_quality_failed": _int(decisions.get("fail")),
        "central_quality_inconclusive": _int(decisions.get("inconclusive")),
        "central_quality_uncertainty_inconclusive": _int(uncertainty.get("inconclusive")),
        "central_quality_missing": count(quality, "campaign_quality_missing_count"),
        "energy_started": count(energy, "measurement_started_count"),
        "energy_verified": count(energy, "measurement_success_count"),
        "energy_failed": count(energy, "measurement_failed_count"),
        "energy_quality_qualified": count(energy, "claim_eligible_count"),
        "generic_excluded": max(0, _int(generic_excluded_count)),
    }
    quality_warning = bool(
        counts["central_quality_accuracy_loss"] or counts["central_quality_failed"] or counts["central_quality_inconclusive"]
        or counts["central_quality_uncertainty_inconclusive"]
        or _int(task_quality.get("fail_count")) or _int(task_quality.get("inconclusive_count"))
    )
    warnings = bool(quality_warning or counts["native_excluded"]
                    or counts["generic_excluded"] or other_warning_count)
    if status == "cancelled":
        label, severity = "Lauf abgebrochen", "cancelled"
    elif status == "failed":
        label, severity = "Lauf fehlgeschlagen", "error"
    elif status == "partial":
        label, severity = "Abgeschlossen mit technischen Teilausfällen", "warning"
    elif status == "ok" and quality_warning:
        label, severity = "Abgeschlossen mit Qualitätswarnungen", "warning"
    elif status == "ok" and warnings:
        label, severity = "Abgeschlossen mit Warnungen", "warning"
    elif status == "ok":
        label, severity = "Abgeschlossen", "success"
    else:
        label, severity = "Laufstatus: " + status, "info"
    details = []
    if counts["native_selected"] is not None:
        details.append(
            f"Native: {counts['native_measured']} erfolgreich, {counts['native_excluded']} ausgeschlossen, "
            f"{counts['native_failed_or_blocked']} fehlgeschlagen/blockiert, {counts['native_missing']} fehlend "
            f"(ausgewählt: {counts['native_selected']})."
        )
    if counts["native_validation_technical_errors"]:
        details.append(f"Zusätzliche Validierungsfehler auf vorhandenen Zeilen: {counts['native_validation_technical_errors']}.")
    if counts["central_quality_completed"] is not None:
        details.append(
            f"Zentrale Qualität: {counts['central_quality_completed']} Auswertungen abgeschlossen, "
            f"{counts['central_quality_cancelled'] or 0} abgebrochen, "
            f"{counts['central_quality_technical_failed'] or 0} technische Fehler; "
            f"{counts['central_quality_reference_close']} referenznah, {counts['central_quality_accuracy_loss']} mit Genauigkeitsverlust; "
            f"davon {counts['central_quality_uncertainty_inconclusive']} statistisch unsicher. "
            f"Legacy: {counts['central_quality_failed']} außerhalb der Grenzen, "
            f"{counts['central_quality_inconclusive']} uneindeutig."
        )
    if counts["central_quality_missing"]:
        details.append(f"Erforderliche Qualitätsresultate fehlend: {counts['central_quality_missing']}.")
    if energy.get("requested") is True:
        accounting = energy.get("accounting") or {}
        details.append(
            f"Native-Energie: {counts['energy_verified']} von {counts['energy_started']} gestarteten Zeilen vollständig verifiziert; "
            f"{counts['energy_failed']} fehlgeschlagen/nicht importierbar."
        )
        if accounting.get("valid_repetition_count") is not None:
            details.append(f"Energie-Replikate: {accounting['valid_repetition_count']}/{accounting.get('requested_repetition_count') if accounting.get('requested_repetition_count') is not None else 'unbekannt'} gültige logische Wiederholungen.")
        if accounting.get("failed_collector_attempt_count") is not None:
            details.append(f"Collectorversuche: {accounting['collector_attempt_count']} gestartet, {accounting['failed_collector_attempt_count']} fehlgeschlagen (separate Versuchszählung).")
    if counts["generic_excluded"]:
        details.append(f"Generische Messmatrix: {counts['generic_excluded']} bekannte Build-Ausschlüsse.")
    return {"status": status, "label": label, "severity": severity,
            "quality_warning": quality_warning, "warning_present": warnings,
            "counts": counts, "detail_lines": details,
            "message": label + (". " + " ".join(details) if details else ""),
            "scientific_pass_semantics": "false means no confirmed release under the declared quality/claim criteria; valid negative observations remain usable descriptively"}


def project_historical_workflow_status(
    source: Mapping[str, Any], native_evidence: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Read-only correction only when archived blockers prove case-local gaps."""
    original = str(source.get("technical_status") or source.get("status") or "unavailable")
    evidence = native_evidence if isinstance(native_evidence, Mapping) else {}
    blockers = source.get("blocking_reasons") or []
    def case_local(reason: Any) -> bool:
        if not isinstance(reason, Mapping):
            return False
        if reason.get("global_integrity_failure") or reason.get("publication_failed"):
            return False
        if reason.get("kind") == "native_evidence":
            return reason.get("stage") in {"run_native_producers", "native_validation", "native_energy"}
        return bool(reason.get("model_id") not in {None, "", "workflow"}
                    and reason.get("stage") in {"build_backend_artifacts", "run_benchmarks", "validate_outputs", "hardware_smoke"})
    applied = bool(original == "failed" and blockers and all(case_local(r) for r in blockers)
                   and _int((evidence.get("runtime") or {}).get("successful_count")) > 0
                   and not source.get("global_integrity_failure") and not source.get("publication_failed"))
    return {"applied": applied, "original_technical_status": original,
            "projected_technical_status": "partial" if applied else original,
            "reason": "archived_case_local_technical_gaps" if applied else "no_proven_case_local_projection",
            "source_mutated": False}


__all__: Sequence[str] = (
    "SCHEMA",
    "SCHEMA_VERSION",
    "blocking_status",
    "workflow_completion_projection",
    "project_historical_workflow_status",
    "derive_native_evidence_status",
    "project_native_evidence_status",
)
